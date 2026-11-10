# omarchy-local-ai

The local-model engine, recipe catalog and internals reference for Omarchy. The Arch package `omarchy-local-ai` installs this tree to `/usr/share/omarchy-local-ai` and symlinks `/usr/bin/omarchy-local-ai` to its executable entry script. Omarchy's thin `omarchy-agent-local` wrapper exposes it as `omarchy agent local <command>`.

The engine resolves recipes, downloads pinned weights, and manages rootless Podman model servers with an internal network, read-only root, dropped capabilities, allowlisted devices and authenticated loopback endpoints. The catalog ships with the engine; it is never fetched at runtime. See [engine contract](docs/engine.md) for the security boundary, lifecycle and state formats.

## Run from a checkout

Python 3.11 or newer is required. On Omarchy, run:

```bash
./omarchy-local-ai list
./omarchy-local-ai use best
./omarchy-local-ai validate .
./tests/engine-test.sh
```

Running models also requires Podman, crun and the Omarchy hardware helpers. Tests use a temporary home and do not start containers. If installed, the real Podman Quadlet generator checks generated units in dry-run mode.

The engine reads the catalog beside `agent_local.py`, following the installed entry script's symlink. Set `OMARCHY_LOCAL_AI_CATALOG` to another catalog directory for development. State and model data live in `~/.local/state/omarchy/agent-local` and `~/.local/share/omarchy/agent-local`.

## DGX Spark without Omarchy

A DGX Spark on NVIDIA's DGX OS (Ubuntu 24.04) runs the engine from a checkout, with nothing of Omarchy installed. The GB10 is detected through `nvidia-smi`. The first `use` asks for sudo once, in a terminal, to install Podman, keep a copy of NVIDIA's CDI spec that Podman 4.9 reads, and let your user linger, the same host setup `link setup` gives a paired Spark:

```bash
git clone https://github.com/omacom/omarchy-local-ai && cd omarchy-local-ai
./omarchy-local-ai list
./omarchy-local-ai use best
./omarchy-local-ai status
```

The model serves an OpenAI-compatible API on loopback (`status` shows the endpoint, from port 11435), authenticated by the key in `~/.config/omarchy/agent-local/key`. Units call back into the checkout, so run `use` again after moving it.

To use the model from another machine, forward its port over SSH and copy the key:

```bash
ssh -N -L 11435:127.0.0.1:11435 spark.local &
scp spark.local:.config/omarchy/agent-local/key ~/spark-key
curl -H "Authorization: Bearer $(cat ~/spark-key)" http://127.0.0.1:11435/v1/models
```

Point any OpenAI-compatible client at `http://127.0.0.1:11435/v1` with that key; `./omarchy-local-ai overlay opencode` on the Spark writes an OpenCode provider configuration to start from (change its key path to the copied key). The endpoint stays on loopback at both ends and only SSH crosses the network.

## Catalog layout

- `catalog.toml` — schema and release version (the single version source).
- `best.toml` — ordered recommendations per hardware class.
- `recipes/<id>.toml` — runtime, pinned Hugging Face artifacts, measured context variants, memory peaks and sampling defaults.
- `runtimes/<id>.toml` — image pinned by digest, input mappings, health probe and allowlisted run options.
- `reviews/` — license and behavior reviews for third-party image digests.
- `measurements/` — evidence for hardware classes a recipe claims.
- `tools/bench` and `bench/context.txt` — checkout-only benchmark script and long prompt.

## Add a recipe

1. Add a recipe and any runtime/review it requires, or add a measured hardware class to an existing recipe. Update `best.toml` where appropriate.
2. Run `./omarchy-local-ai validate .`, then `./omarchy-local-ai use <id>` on the target hardware.
3. Run `./tools/bench <id> --output measurements/<id>-<gpu>-<ram>.toml` and reference the measurement in the recipe.
4. Submit the recipe, review and measurement together. CI validates the catalog online and runs engine tests. Merging catalog changes grants software publication authority within the engine's security boundary.

## Releases and packaging

Update `version` in `catalog.toml`, validate and test, then tag the release `vX.Y.Z` matching that version. The `omarchy-pkgs` recipe builds the Arch package from that tag, installing the engine, catalog and docs together. Packaging is maintained in `omarchy-pkgs`; this repository is its source tree.

`bench/` and `tools/` do not need to ship in the Arch package. Measurement records still ship: recipe validation and the panel use them.

After package updates, `omarchy-agent-local refresh --quiet` rewrites the panel's list and status caches. It does not download a catalog or change running models.
