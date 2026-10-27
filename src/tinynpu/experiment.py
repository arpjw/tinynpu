"""Freeze, evaluate, and package the fixed TinyNPU experiment."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits

from tinynpu.evidence import ROOT, environment, sha256, source_fingerprint, write_json
from tinynpu.inference import infer_hardware, transfer_cost
from tinynpu.model import clipping_count, load_model, prepare_model
from tinynpu.simulator import DEFAULT_BINARY, Simulator


def read_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text())
    expected = {
        "hardware": {
            "rows": 4,
            "columns": 4,
            "operand_bits": 8,
            "accumulator_bits": 32,
            "max_reduction_length": 64,
            "dataflow": "output_stationary",
        },
        "model": {
            "dataset": "sklearn_digits",
            "layers": [64, 16, 10],
            "activation": "relu",
            "train_fraction": 0.70,
            "calibration_fraction": 0.10,
            "test_fraction": 0.20,
        },
        "experiment": {
            "batch_sizes": [1, 4],
            "transfer_bytes_per_cycle": [1, 4, 8],
            "transfer_overlap": False,
            "randomized_rtl_cases": 1024,
            "trace_schema_version": 1,
        },
    }
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema.")
    if set(config) != {"schema_version", "seed", "hardware", "model", "experiment"}:
        raise ValueError("Unknown top-level configuration fields.")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported {key} settings; update the protocol explicitly.")
    return config


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output: Path, config_path: Path) -> dict[str, Any]:
    config = read_config(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    initial_source = source_fingerprint()
    source_id = hashlib.sha256(json.dumps(initial_source, sort_keys=True).encode()).hexdigest()
    prediction_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, int]] = []
    batch_rows: list[dict[str, Any]] = []
    try:
        # The hardware correctness gate precedes model preparation and evaluation.
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_rtl.py"),
                "--output",
                str(output / "rtl-verification.json"),
                "--cases",
                str(config["experiment"]["randomized_rtl_cases"]),
                "--seed",
                str(config["seed"]),
            ],
            check=True,
            cwd=ROOT,
        )
        print("RTL campaign passed. Preparing the fixed model.", flush=True)
        paths = prepare_model(output, seed=config["seed"])
        model = load_model(paths["model"])
        qualification = json.loads((output / "rtl-verification.json").read_text())
        frozen_binary = output / "Varray"
        shutil.copy2(DEFAULT_BINARY, frozen_binary)
        binary_hash = sha256(frozen_binary)
        if (
            qualification["status"] != "passed"
            or qualification["passed_cases"] != config["experiment"]["randomized_rtl_cases"]
            or qualification["seed"] != config["seed"]
            or qualification["array_binary_sha256"] != binary_hash
            or any(
                initial_source.get(name) != digest
                for name, digest in qualification["source_sha256"].items()
            )
            or source_fingerprint() != initial_source
        ):
            raise RuntimeError("RTL qualification does not match the frozen source and executable.")
        shutil.copyfile(config_path, output / "config.toml")
        frozen_paths = {**paths, "verification": output / "rtl-verification.json"}
        frozen_hashes = {name: sha256(path) for name, path in frozen_paths.items()}
        freeze = {
            "schema_version": 1,
            "frozen_at": datetime.now(UTC).isoformat(),
            "held_out_evaluated": False,
            "config": config,
            "config_sha256": sha256(config_path),
            "source_id": source_id,
            "source_files": initial_source,
            "environment": environment(),
            "process_argv": [sys.executable, *sys.argv],
            "resolved_config": str(config_path),
            "resolved_output": str(output),
            "reproduction_command": [
                "make",
                "experiment",
                f"RUN_DIR={output}-reproduction",
                f"CONFIG={config_path}",
            ],
            "input_sha256": frozen_hashes,
            "simulator_binary_sha256": binary_hash,
            "evaluated_executable": "Varray",
        }
        write_json(output / "freeze.json", freeze)
        print("Model, splits, settings, and source frozen. Evaluating held-out images.", flush=True)
        with np.load(paths["splits"], allow_pickle=False) as splits:
            indices = splits["test_indices"]
        x_all, y_all = load_digits(return_X_y=True)
        x, y = x_all[indices], y_all[indices]
        reference = model.infer_software(x)
        fp_predictions = model.predict_float(x)
        examples: list[dict[str, Any]] = []
        checked_tiles = 0
        checked_values = 0
        with Simulator(frozen_binary) as simulator:
            for batch_size in config["experiment"]["batch_sizes"]:
                predictions: list[int] = []
                for start in range(0, len(x), batch_size):
                    stop = min(start + batch_size, len(x))
                    # Fixed test offsets provide representative, unselected traces.
                    capture = start in (0, 4)
                    result = infer_hardware(
                        model, x[start:stop], simulator, trace=capture, job_id=start
                    )
                    np.testing.assert_array_equal(
                        result.layer1_acc, reference.layer1_acc[start:stop]
                    )
                    np.testing.assert_array_equal(
                        result.layer2_acc, reference.layer2_acc[start:stop]
                    )
                    np.testing.assert_array_equal(
                        result.predictions, reference.predictions[start:stop]
                    )
                    checked_tiles += len(result.tiles)
                    checked_values += result.layer1_acc.size + result.layer2_acc.size
                    predictions.extend(int(p) for p in result.predictions)
                    for tile in result.tiles:
                        tile_rows.append({"batch_size": batch_size, **tile})
                    for bandwidth in config["experiment"]["transfer_bytes_per_cycle"]:
                        costs = [
                            transfer_cost(t["m"], t["k"], t["n"], bandwidth) for t in result.tiles
                        ]
                        counters = {
                            key: sum(t[key] for t in result.tiles)
                            for key in (
                                "compute_cycles",
                                "pause_cycles",
                                "clear_cycles",
                                "valid_macs",
                                "useful_macs",
                                "padded_macs",
                            )
                        }
                        transfers = {
                            key: sum(t[key] for t in costs)
                            for key in ("load_bytes", "store_bytes", "load_cycles", "store_cycles")
                        }
                        batch_rows.append(
                            {
                                "batch_size": batch_size,
                                "bandwidth": bandwidth,
                                "batch_start": start,
                                "images": stop - start,
                                **counters,
                                **transfers,
                                "total_cycles": (
                                    counters["compute_cycles"]
                                    + counters["pause_cycles"]
                                    + counters["clear_cycles"]
                                    + transfers["load_cycles"]
                                    + transfers["store_cycles"]
                                ),
                            }
                        )
                    if capture:
                        examples.append(
                            {
                                "id": int(indices[start]),
                                "label": int(y[start]),
                                "prediction": int(result.predictions[0]),
                                "pixels": x[start].tolist(),
                                "batch_size": batch_size,
                                "batch_labels": y[start:stop].tolist(),
                                "batch_predictions": result.predictions.tolist(),
                                "traces": result.traces,
                                "logits": result.logits[0].tolist(),
                            }
                        )
                for offset, prediction in enumerate(predictions):
                    prediction_rows.append(
                        {
                            "batch_size": batch_size,
                            "dataset_index": int(indices[offset]),
                            "label": int(y[offset]),
                            "float_prediction": int(fp_predictions[offset]),
                            "quantized_prediction": int(reference.predictions[offset]),
                            "rtl_prediction": prediction,
                        }
                    )
                print(
                    f"Batch {batch_size}: {len(predictions)} images verified exactly.", flush=True
                )

        comparisons = []
        for batch_size in config["experiment"]["batch_sizes"]:
            for bandwidth in config["experiment"]["transfer_bytes_per_cycle"]:
                rows = [
                    r
                    for r in batch_rows
                    if r["batch_size"] == batch_size and r["bandwidth"] == bandwidth
                ]
                images = sum(r["images"] for r in rows)
                compute = sum(r["compute_cycles"] + r["pause_cycles"] for r in rows)
                total = sum(r["total_cycles"] for r in rows)
                useful = sum(r["useful_macs"] for r in rows)
                if useful != images * 1184:
                    raise AssertionError("Useful MAC accounting does not match the network.")
                comparisons.append(
                    {
                        "batch_size": batch_size,
                        "bandwidth": bandwidth,
                        "images": images,
                        "batches": len(rows),
                        "compute_cycles": compute,
                        "total_cycles": total,
                        "useful_macs": useful,
                        "compute_cycles_per_image": compute / images,
                        "total_cycles_per_image": total / images,
                        "mean_batch_cycles": total / len(rows),
                        "compute_utilization": useful / (16 * compute),
                        "total_utilization": useful / (16 * total),
                    }
                )
        if source_fingerprint() != initial_source:
            raise RuntimeError("Source changed during the frozen experiment.")
        if {name: sha256(path) for name, path in frozen_paths.items()} != frozen_hashes:
            raise RuntimeError("Frozen model, split, training, or verification artifacts changed.")
        if sha256(config_path) != freeze["config_sha256"]:
            raise RuntimeError("Configuration changed during the frozen experiment.")
        if sha256(frozen_binary) != freeze["simulator_binary_sha256"]:
            raise RuntimeError("Simulator binary changed during the frozen experiment.")
        summary = {
            "schema_version": 1,
            "source_id": source_id,
            "test_count": len(x),
            "float_correct": int(np.count_nonzero(fp_predictions == y)),
            "quantized_correct": int(np.count_nonzero(reference.predictions == y)),
            "float_accuracy": float(np.mean(fp_predictions == y)),
            "quantized_accuracy": float(np.mean(reference.predictions == y)),
            "rtl_mismatches": 0,
            "verified_tiles": checked_tiles,
            "verified_accumulator_values": checked_values,
            "test_input_clipped_values": clipping_count(x, model.input_scale),
            "test_hidden_clipped_values": clipping_count(
                model.hidden_float_from_acc(reference.layer1_acc), model.hidden_scale
            ),
            "source_unchanged": True,
            "frozen_inputs_unchanged": True,
            "limitations": [
                "RTL simulation, not a physical chip.",
                "Transfer costs modeled with serialized per-tile reloads and ideal local buffer.",
                "Bias, ReLU, requantization and argmax run on host; host time excluded.",
                "Batch arrival/wait time excluded; throughput is not single-request latency.",
                "One small dataset and one fixed model; no general performance claim.",
            ],
        }
        write_csv(output / "tiles.csv", tile_rows)
        write_csv(output / "batches.csv", batch_rows)
        write_csv(output / "predictions.csv", prediction_rows)
        write_csv(output / "comparisons.csv", comparisons)
        write_json(output / "summary.json", summary)
        write_json(
            output / "demo.json",
            {
                "schema_version": 1,
                "summary": summary,
                "comparisons": comparisons,
                "examples": examples,
            },
        )
        write_json(
            output / "manifest.json",
            {
                "schema_version": 1,
                "completed_at": datetime.now(UTC).isoformat(),
                "source_id": source_id,
                "status": "passed",
                "files": {
                    p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in sorted(output.iterdir())
                    if p.is_file()
                },
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as error:
        for name, rows in (
            ("partial-tiles.csv", tile_rows),
            ("partial-batches.csv", batch_rows),
            ("partial-predictions.csv", prediction_rows),
        ):
            if rows:
                write_csv(output / name, rows)
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "source_id": source_id,
                "diagnostics": getattr(error, "diagnostics", None),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "run")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.toml")
    args = parser.parse_args()
    run_experiment(args.output.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
