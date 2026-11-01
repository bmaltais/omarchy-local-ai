# omarchy-local-ai

The catalog behind `omarchy agent local use best`: which local model to run on which machine, on which runtime, with which weights, and the measurements that justify each choice.

Omarchy owns the mechanism and the security boundary (an internal network with no egress, read-only root, dropped capabilities, allowlisted devices, the key as a Podman secret, typed unit generation, post-start auth checks). This repo owns the choices. Anyone who can merge here is a software publisher for every Omarchy machine that runs a local model, so merges are releases.

## Layout

- `catalog.toml` — the schema version. Omarchy reads the `schema-<n>` branch for the schema it understands.
- `best.toml` — the ordered recommendation per hardware class (GPU family and RAM class).
- `recipes/<id>.toml` — one model setup: runtime, pinned Hugging Face artifacts, measured context variants and peak memory, sampling defaults, extra server arguments.
- `runtimes/<id>.toml` — one backend image pinned by digest, how to pass the standard inputs to it, its health probe, and the allowlisted run options it needs.
- `reviews/` — the license and behavior review for each third-party image digest.
- `measurements/` — the evidence for every hardware class a recipe claims.
- `bench/context.txt` — the long prompt `omarchy agent local bench` measures with, pinned here so numbers are comparable across machines.

## Adding a recipe or a hardware class

1. Write the recipe, or add a `[[hardware]]` entry for your class.
2. Run it from your fork: point `~/.config/omarchy/agent-local/catalog.toml` at it (`url`, `ref`), then `omarchy agent local use <id>`.
3. `omarchy agent local bench --output measurements/<id>-<gpu>-<ram>.toml`.
4. Open a pull request with the recipe and the measurement. CI runs `omarchy-agent-local-validate --online`.
