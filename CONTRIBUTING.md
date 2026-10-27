# Contributing

Start with [README.md](README.md), [project status](docs/STATUS.md),
[AGENTS.md](AGENTS.md), and the [architecture contract](docs/ARCHITECTURE.md).
Keep changes focused on the fixed 4×4 INT8/INT32 lab; explain any proposed
change to its arithmetic, timing, or experiment protocol before implementation.

Run `make setup` once and `make check` before submitting a change. For viewer
changes, also run `node --check viewer/app.js`, serve the project with
`make serve`, and exercise digit selection, batch switching, stepping, playback,
and missing-data behavior in a browser. Generate local evidence with
`make experiment` when needed; choose a new `RUN_DIR` to preserve earlier runs.

Describe the problem, resulting behavior, and validation in the pull request.
Add focused tests for arithmetic or protocol changes. Keep the Python oracle
independent of the RTL implementation, and never tune against the held-out split.
Distinguish measured RTL cycles from modeled transfers and physical performance.

Keep generated models, datasets, binaries, traces, and local reports in ignored
`artifacts/` or `build/`. Do not commit credentials or environment files.
Update documentation when behavior changes, and retain attribution when using
third-party material. Contributions to original project code are under the
[MIT license](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md) for data.
