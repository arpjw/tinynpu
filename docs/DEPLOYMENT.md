# Static deployment

Production viewer: https://tinynpu.vercel.app

The alias https://tinynpu-lab.vercel.app serves the same deployment.

The hosted demo replays recorded RTL states. It does not run Verilator, train a
model, or accept inference jobs in the browser. No backend or secrets are needed.

Generate a new experiment and export it:

```sh
make setup
make check
make experiment RUN_DIR=artifacts/release
make site RUN_DIR=artifacts/release
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8766. The exporter verifies every file in the completed
experiment manifest, checks source identifiers and the demo summary, then writes
an allowlist of eight deployment files to ignored `dist/`. It refuses an export folder
with unexpected files. JSON whitespace is compacted without changing trace values.

`provenance.json` records the experiment source fingerprint, original demo hash,
and hashes of the deployed assets. The complete evidence stays in `artifacts/`;
the site excludes model weights, binaries, training data, and environment paths.
Digit pixels are covered by the dataset attribution shipped with the site.

## Vercel

Deploy only `dist/`, after checking its files and local playback. Authenticate
with the Vercel CLI, create a project in your own account, and link explicitly:

```sh
cd dist
vercel link --project YOUR_PROJECT --scope YOUR_TEAM --yes
vercel project inspect --non-interactive
vercel deploy --prod
```

Verify the production URL and playback in an unauthenticated browser. Compare
the downloaded `provenance.json` and served asset hashes to the local export.
`vercel.json` is a server configuration input: its hash is recorded, but Vercel
does not serve it as a public asset.
The included `vercel.json` serves plain static files with no build or install
step. The TinyNPU deployment is an explicit export, with no Git-triggered deploy
configured: pushing source does not silently replace the frozen demo.

Some CLI versions create an ignored `.env.local` when linking. Keep that file
outside the export directory before rebuilding; the exporter deliberately rejects
unexpected files. `vercel deploy --dry --json` lists the exact upload inputs.

To update it, use a fresh run directory and repeat verification and export.
Historical experiment fingerprints can differ from later viewer sources; asset
hashes describe the actual published viewer separately.
