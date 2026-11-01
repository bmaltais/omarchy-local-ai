# omarchy-local-ai

The local-model engine, recipe catalog and internals reference for Omarchy. The Arch package `omarchy-local-ai` installs this tree to `/usr/share/omarchy-local-ai` and symlinks `/usr/bin/omarchy-local-ai` to its executable entry script. Omarchy's thin `omarchy-agent-local` wrapper exposes it as `omarchy agent local <command>`.

The engine resolves recipes, downloads pinned weights, and manages rootless Podman model servers with an internal network, read-only root, dropped capabilities, allowlisted devices and authenticated loopback endpoints. The catalog ships with the engine; it is never fetched at runtime. See [engine internals](docs/engine.md) for the security boundary, lifecycle and state formats.

## Run from a checkout

Python 3.11 or newer is required. On Omarchy, run:

```bash
./omarchy-local-ai list
./omarchy-local-ai use best
./omarchy-local-ai validate .
./tests/engine-test.sh
```

Running models also requires Podman, crun and the Omarchy hardware helpers. Tests use a temporary home and do not start containers. If installed, the real Podman Quadlet generator checks generated units in dry-run mode.

The engine reads the catalog beside `agent_local.py`, following the installed entry script's symlink. Set `OMARCHY_LOCAL_AI_CATALOG` to another catalog directory for development. User overrides remain in `~/.config/omarchy/agent-local/{recipes,runtimes}`; state and model data remain in `~/.local/state/omarchy/agent-local` and `~/.local/share/omarchy/agent-local`.

## Catalog layout

- `catalog.toml` — schema and release version (the single version source).
- `best.toml` — ordered recommendations per hardware class.
- `recipes/<id>.toml` — runtime, pinned Hugging Face artifacts, measured context variants, memory peaks and sampling defaults.
- `runtimes/<id>.toml` — image pinned by digest, input mappings, health probe and allowlisted run options.
- `reviews/` — license and behavior reviews for third-party image digests.
- `measurements/` — evidence for hardware classes a recipe claims.
- `bench/context.txt` — shared long benchmark prompt.

## Add a recipe

1. Add a recipe and any runtime/review it requires, or add a measured hardware class to an existing recipe. Update `best.toml` where appropriate.
2. Run `./omarchy-local-ai validate .`, then `./omarchy-local-ai use <id>` on the target hardware.
3. Run `./omarchy-local-ai bench <id> --output measurements/<id>-<gpu>-<ram>.toml` and reference the measurement in the recipe.
4. Submit the recipe, review and measurement together. CI validates the catalog online and runs engine tests. Merging catalog changes grants software publication authority within the engine's security boundary.

## Releases and packaging

Update `version` in `catalog.toml`, validate and test, then tag the release `vX.Y.Z` matching that version. The `omarchy-pkgs` recipe builds the Arch package from that tag, installing the engine, catalog and docs together. Packaging is maintained in `omarchy-pkgs`; this repository is its source tree.

After package updates, `omarchy-agent-local refresh --quiet` rewrites the panel's list and status caches. It does not download a catalog or change running models.
