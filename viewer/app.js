"use strict";

(() => {
  const element = (id) => document.getElementById(id);
  const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Unavailable";
  const number = (value) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "Unavailable";
  const cells = [];
  let data;
  let current;
  let frameIndex = 0;
  let timer = null;

  function pause() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    element("play").textContent = "Play";
    element("play").setAttribute("aria-pressed", "false");
  }

  function renderFrame() {
    const trace = current.traces[frameIndex];
    element("scrubber").value = String(frameIndex);
    element("scrubber").setAttribute("aria-valuetext", `Frame ${frameIndex + 1} of ${current.traces.length}, layer ${trace.layer ?? "unspecified"}, cycle ${trace.cycle}`);
    element("layer").textContent = `Layer ${trace.layer ?? "–"}`;
    element("tile").textContent = `Tile ${trace.tile_id}`;
    element("cycle").textContent = `Cycle ${trace.cycle}`;
    element("frame").textContent = `Frame ${frameIndex + 1} / ${current.traces.length}`;
    element("phase").textContent = trace.phase === "clear" ? "Clear accumulators" : `Step ${trace.logical_step}${trace.enable ? "" : " · paused"}`;
    element("step").disabled = frameIndex === current.traces.length - 1;
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        const pe = trace.pes[row][col];
        const cell = cells[row * 4 + col];
        const active = Boolean(pe.active);
        const useful = active && Boolean(pe.useful);
        const padded = active && Boolean(pe.padded);
        const kind = useful ? "useful MAC" : padded ? "padding MAC" : active ? "active MAC" : "idle / forwarding";
        cell.root.className = `pe${useful ? " active" : padded ? " padded" : ""}`;
        cell.state.textContent = useful ? "MAC" : padded ? "PAD" : "·";
        cell.accumulator.textContent = number(pe.accumulator);
        const a = pe.a_valid ? pe.a : "–";
        const b = pe.b_valid ? pe.b : "–";
        cell.operands.textContent = `${a} × ${b}`;
        cell.root.setAttribute("aria-label", `Processing element ${row}, ${col}: ${kind}, A ${a}, B ${b}, accumulator ${pe.accumulator}`);
      }
    }
  }

  function start() {
    if (current.traces.length < 2) return;
    if (frameIndex >= current.traces.length - 1) frameIndex = 0;
    element("play").textContent = "Pause";
    element("play").setAttribute("aria-pressed", "true");
    renderFrame();
    timer = setInterval(() => {
      frameIndex += 1;
      renderFrame();
      if (frameIndex >= current.traces.length - 1) pause();
    }, Number(element("speed").value));
  }

  function drawDigit(example) {
    const canvas = element("digit");
    const ctx = canvas.getContext("2d");
    if (ctx) {
      for (let i = 0; i < 64; i += 1) {
        const intensity = Math.max(0, Math.min(1, example.pixels[i] / 16));
        const shade = Math.round(245 - intensity * 210);
        ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
        ctx.fillRect((i % 8) * 20, Math.floor(i / 8) * 20, 20, 20);
      }
    }
    canvas.setAttribute("aria-label", `Handwritten digit ${example.label}, predicted ${example.prediction}`);
  }

  function selectExample(index) {
    pause();
    current = data.examples[index];
    frameIndex = 0;
    element("example").value = String(index);
    element("prediction").textContent = String(current.prediction);
    element("label").textContent = String(current.label);
    element("scrubber").max = String(current.traces.length - 1);
    element("play").disabled = current.traces.length < 2;
    element("batch-detail").textContent = current.batch_size > 1
      ? `Batch labels: ${current.batch_labels.join(", ")} · Predictions: ${current.batch_predictions.join(", ")}`
      : "";
    drawDigit(current);
    renderFrame();
  }

  function setBatch() {
    const previousId = current?.id;
    const batch = Number(element("batch").value);
    const available = data.examples.map((example, index) => ({ example, index })).filter(({ example }) => example.batch_size === batch);
    element("example").replaceChildren();
    for (const { example, index } of available) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `Digit ${example.label} · #${example.id}`;
      element("example").append(option);
    }
    const selected = available.find(({ example }) => example.id === previousId) ?? available[0];
    selectExample(selected.index);
  }

  function createCells() {
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        const root = document.createElement("div");
        root.className = "pe";
        root.setAttribute("role", "img");
        const heading = document.createElement("div");
        heading.className = "pe-id";
        const label = document.createElement("span");
        label.textContent = `PE ${row},${col}`;
        const state = document.createElement("span");
        heading.append(label, state);
        const accumulator = document.createElement("div");
        accumulator.className = "pe-acc";
        const operands = document.createElement("div");
        operands.className = "pe-operands";
        root.append(heading, accumulator, operands);
        element("array").append(root);
        cells.push({ root, accumulator, operands, state });
      }
    }
  }

  function validate(payload) {
    if (payload.schema_version !== 1) throw new Error("This viewer requires demo schema version 1.");
    if (!payload.summary || !Array.isArray(payload.comparisons)) throw new Error("The artifact is missing its summary or comparison measurements.");
    if (!Array.isArray(payload.examples) || payload.examples.length === 0) throw new Error("No recorded examples were found in this artifact.");
    for (const example of payload.examples) {
      if (![1, 4].includes(example.batch_size) || !Array.isArray(example.pixels) || example.pixels.length !== 64 || !example.pixels.every(Number.isFinite)) throw new Error("An example has invalid batch or pixel data.");
      if (!Array.isArray(example.batch_labels) || !Array.isArray(example.batch_predictions)) throw new Error("An example is missing its batch labels or predictions.");
      if (!Array.isArray(example.traces) || example.traces.length === 0) throw new Error("An example has no recorded RTL frames.");
      for (const trace of example.traces) {
        if (trace.schema_version !== 1 || !Array.isArray(trace.pes) || trace.pes.length !== 4 || !trace.pes.every((row) => Array.isArray(row) && row.length === 4 && row.every((pe) => pe && Number.isFinite(pe.a) && Number.isFinite(pe.b) && Number.isFinite(pe.accumulator)))) throw new Error("A recorded frame has invalid processing-element data.");
      }
    }
  }

  function renderEvidence() {
    element("float-accuracy").textContent = percent(data.summary.float_accuracy);
    element("quant-accuracy").textContent = percent(data.summary.quantized_accuracy);
    element("mismatches").textContent = number(data.summary.rtl_mismatches);
    element("test-count").textContent = number(data.summary.test_count);
    element("source").textContent = `Source: ${data.summary.source_id ?? "not recorded"}`;
    for (const row of data.comparisons) {
      const tr = document.createElement("tr");
      for (const value of [number(row.batch_size), number(row.bandwidth), number(row.compute_cycles_per_image), number(row.total_cycles_per_image), percent(row.compute_utilization), percent(row.total_utilization)]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      element("comparisons").append(tr);
    }
  }

  element("play").addEventListener("click", () => timer === null ? start() : pause());
  element("reset").addEventListener("click", () => { pause(); frameIndex = 0; renderFrame(); });
  element("step").addEventListener("click", () => { pause(); frameIndex = Math.min(frameIndex + 1, current.traces.length - 1); renderFrame(); });
  element("scrubber").addEventListener("input", (event) => { pause(); frameIndex = Number(event.target.value); renderFrame(); });
  element("speed").addEventListener("change", () => { if (timer !== null) { pause(); start(); } });
  element("batch").addEventListener("change", setBatch);
  element("example").addEventListener("change", (event) => selectExample(Number(event.target.value)));
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });

  async function load() {
    try {
      const path = new URLSearchParams(window.location.search).get("data") || document.querySelector('meta[name="trace-data"]').content;
      const url = new URL(path, window.location.href);
      if (url.origin !== window.location.origin) throw new Error("Trace artifacts must be served from the same origin as the viewer.");
      const response = await fetch(url);
      if (!response.ok) throw new Error(`The trace artifact returned HTTP ${response.status}.`);
      data = await response.json();
      validate(data);
      for (const option of element("batch").options) option.disabled = !data.examples.some((example) => example.batch_size === Number(option.value));
      element("batch").value = String(data.examples[0].batch_size);
      createCells();
      renderEvidence();
      setBatch();
      element("status").hidden = true;
      element("content").hidden = false;
    } catch (error) {
      element("status").className = "status error";
      element("status").setAttribute("role", "alert");
      element("status").textContent = `No trace to display. ${error instanceof Error ? error.message : String(error)} Generate the demo artifact, serve the repository over HTTP, and reload. No simulated activity is substituted for missing data.`;
    }
  }

  load();
})();
