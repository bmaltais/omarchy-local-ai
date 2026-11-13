# omarchy-local-ai (bmaltais fork)

Fork of omacom/omarchy-local-ai that startdgx pins and installs on DGX Spark hosts. Engine code runs as root on those hosts.

## Remotes and branches
- `origin` is upstream (omacom): fetch and merge only; a push there is refused. Push to `fork`, always by name (`git push fork <branch>`).
- `remote-machines` is the fork's working branch and the base for PRs and tags. `fork/master` is old on purpose; don't base on it.
- Merge `origin/master` into `remote-machines`; don't rebase (the dgx tags point at its commits).

## Checks
- `tests/engine-test.sh` and `./omarchy-local-ai validate .` before every commit. Fix every warning. Tests fake the runtime and never touch a machine or ssh.
- Never run the engine against a real machine from here.

## Releases
A tag is `vX.Y.Z-dgx.N`, where `vX.Y.Z` is the upstream release the branch contains. Cut it with startdgx's `scripts/release-engine` and follow startdgx's `docs/engine-releases.md`; startdgx then bumps `ENGINE_TAG`.
