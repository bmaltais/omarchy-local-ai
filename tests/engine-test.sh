#!/bin/bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export HOME="$scratch" ENGINE_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME="$HOME/.config"

python3 -I <<'PY'
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.environ["ENGINE_ROOT"])
import agent_local as a
import runpy
bench = SimpleNamespace(**runpy.run_path(str(Path(os.environ["ENGINE_ROOT"]) / "tools/bench")))


def check(name, condition, detail=""):
  if not condition:
    print(f"not ok - {name}", file=sys.stderr)
    if detail:
      print(detail, file=sys.stderr)
    raise SystemExit(1)
  print(f"ok - {name}", flush=True)


def skip(name):
  print(f"ok - {name} # SKIP", flush=True)


def raises(name, fn, message=""):
  try:
    fn()
  except a.LocalError as error:
    check(name, message in str(error), str(error))
    return
  check(name, False, "expected LocalError")


runtime = {
  "image": "registry.example/test/server@sha256:" + "a" * 64,
  "tier": "third-party", "review": "reviews/server.toml", "port": 8080,
  "health": {"probe": ["curl", "http://localhost:8080/health"]},
  "command": ["server"], "reserved": ["--host", "API_KEY"],
  "inputs": {"model": {"arg": "--model"}, "draft": {"arg": "--draft"},
             "ctx": {"arg": "--ctx"}, "alias": {"arg": "--alias"}, "key": {"env": "API_KEY"}},
}
variant = {"ctx": 4096, "peak_ram_gib": 16, "peak_gpu_gib": 16}
artifact = {"repo": "test/model", "revision": "b" * 40, "license": "mit",
            "files": ["model.gguf", "tokenizer/*.json"], "model": "model.gguf"}
recipe = {"name": "Test model", "summary": "Fixture", "runtime": "server",
          "hardware": [{"gpu": "amd-strix-halo", "ram_class": 64, "measured": "measurements/model.toml"}],
          "artifacts": [artifact], "variant": [variant]}
gpu = {"vendor": "amd", "family": "amd-strix-halo", "memory_gib": 48}
profile = {"platform": "x86", "ram_class": 64, "ram_gib": 64, "gpus": [gpu]}
entry = {"gpu": gpu["family"], "ram_class": 64, "recipes": ["model"]}
check("valid recipe and runtime fixtures are accepted", not a.validate_runtime("server", runtime) and not a.validate_recipe("model", recipe, {"server": runtime}))

for rid in ("..", "a..b", "-a", "a-", ".a", "a."):
  check(f"recipe and runtime reject unsafe id {rid}", bool(a.validate_runtime(rid, runtime)) and bool(a.validate_recipe(rid, recipe, {"server": runtime})))
for name, mutate, fragment in (
  ("reserved input target", lambda r: r["inputs"].update(model={"arg": "--host"}), "targets reserved"),
  ("duplicate input target", lambda r: r["inputs"].update(alias={"arg": "--model"}), "same target"),
  ("host IPC", lambda r: r.update(run={"ipc": "host"}), "not an allowed"),
  ("privileged container", lambda r: r.update(run={"privileged": True}), "not an allowed"),
  ("command line key without opt-in", lambda r: r["inputs"].update(key={"arg": "--key"}), "key_on_cmdline = true"),
  ("a positional input other than the model", lambda r: r["inputs"].update(alias={"positional": True}), "only the model can be positional"),
  ("command line key without shell", lambda r: (r.update(key_on_cmdline=True), r["inputs"].update(key={"arg": "--key"})), "needs shell"),
):
  changed = copy.deepcopy(runtime)
  mutate(changed)
  check(f"runtime rejects {name}", any(fragment in e for e in a.validate_runtime("server", changed)))
# A key may use its own reserved environment variable, as in the valid fixture.
changed = copy.deepcopy(runtime)
changed.update(key_on_cmdline=True, shell="/bin/bash")
changed["inputs"]["key"] = {"arg": "--host"}
check("key cannot commandeer a reserved argument", any("reserved" in e for e in a.validate_runtime("server", changed)))

for flag in ("--host", "--host=0.0.0.0", "--model", "--model=other"):
  changed = copy.deepcopy(recipe)
  changed["server"] = {"args": [flag]}
  check(f"server args cannot override {flag}", any("server.args" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
for name, args, fragment in (("a reserved flag", ["--host"], "reserved"), ("an input's flag", ["--model=other"], "inputs"),
                             ("a shared flag", ["--threads", "2"], "already in server.args")):
  changed = copy.deepcopy(recipe)
  changed["server"] = {"args": ["--threads", "4"]}
  changed["variant"] = [dict(variant, args=args)]
  check(f"variant args reject {name}", any(".variant[0].args" in e and fragment in e for e in a.validate_recipe("model", changed, {"server": runtime})))
for field, value in (("files", ["../weights"]), ("files", ["/weights"]), ("dest", "../outside"), ("dest", "/outside")):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0][field] = value
  check(f"artifact rejects escaping {field} {value}", any("relative path" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
for role in ("model", "draft"):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0][role] = "model.gguf"
  changed["artifacts"].append(copy.deepcopy(changed["artifacts"][0]))
  check(f"duplicate {role} role is rejected", any(f".{role}: already named" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
check("identical best selectors are rejected", any("same selectors" in e for e in a.validate_best({"class": [entry, entry]}, {"model": recipe})))
check("best cannot recommend an unmeasured class", any("no measured" in e for e in a.validate_best({"class": [dict(entry, ram_class=128)]}, {"model": recipe})))

catalog = SimpleNamespace(recipes={"model": copy.deepcopy(recipe)}, runtimes={"server": runtime}, best={"class": [entry]})
check("best requires an exact GPU and RAM pair", a.match_class({"class": [entry]}, profile, dict(gpu, family="other")) is None and a.match_class({"class": [entry]}, dict(profile, ram_class=128), gpu) is None)
check("best requires a GPU selector", any(".gpu" in e for e in a.validate_best({"class": [{"ram_class": 64, "recipes": ["model"]}]}, {"model": recipe})))
check("best rejects platform selectors", any(".platform" in e for e in a.validate_best({"class": [dict(entry, platform="x86")]}, {"model": recipe})))
catalog.recipes["model"]["variant"] = [variant, dict(variant, ctx=8192), dict(variant, ctx=16384, peak_ram_gib=60)]
check("largest fitting context wins", a.resolve_best(catalog, profile, gpu)[1]["ctx"] == 8192)
small = dict(variant, ctx=8192, args=["--parallel", "1"])
catalog.recipes["model"]["variant"] = [small, dict(variant, ctx=16384, peak_ram_gib=60, args=["--parallel", "3"])]
picked = a.resolve_best(catalog, profile, gpu)[1]
check("a smaller machine falls back to the variant sized for it", picked is small)
catalog.recipes["model"]["variant"] = [dict(small, ctx=16384, peak_ram_gib=40), dict(small, ctx=16384, peak_ram_gib=60, args=["--parallel", "3"])]
check("at the same context a smaller machine takes the smaller setup", a.resolve_best(catalog, profile, gpu)[1]["peak_ram_gib"] == 40)
check("at the same context a larger machine takes the larger setup", a.resolve_best(catalog, dict(profile, ram_gib=128), dict(gpu, memory_gib=96))[1]["args"] == ["--parallel", "3"])
_, argv, _, _ = a.server_invocation(runtime, dict(recipe, server={"args": ["--vision"]}), picked, {"model": "w"}, "model")
check("a variant's args follow the shared server args", argv[-3:] == ["--vision", "--parallel", "1"])
for reason, bad in (("non-fitting variants", dict(recipe, variant=[dict(variant, peak_gpu_gib=100)])),
                    ("unmeasured hardware", dict(recipe, hardware=[dict(recipe["hardware"][0], ram_class=128)]))):
  catalog.recipes["first"] = bad
  catalog.best = {"class": [dict(entry, recipes=["first", "model"])]}
  result = a.resolve_best(catalog, profile, gpu)
  check(f"best falls through {reason}", result[0] == "model" and result[2][0][0] == "first")
catalog.recipes["first"] = recipe
catalog.recipes["model"]["variant"] = [variant, dict(variant, ctx=8192)]
result = a.resolve_best(catalog, profile, gpu, ctx=8192)
check("best falls through a recipe without the requested context", result[0] == "model" and result[1]["ctx"] == 8192 and "ctx 8192" in result[2][0][1])
catalog.best = {"class": []}
try:
  a.resolve_best(catalog, profile, gpu)
except a.LocalError as error:
  check("missing recommendation reports the profile", all(str(v) in str(error) for v in (gpu["family"], 64, "x86")))
else:
  check("missing recommendation raises", False)
rival = dict(gpu, family="amd-other")
tied = dict(profile, gpus=[gpu, rival])
raises("different GPU families tied for memory require selection", lambda: a.choose_gpu(tied), "--gpu")
check("explicit GPU resolves a memory tie", a.choose_gpu(tied, "amd-other") == rival)
with patch.object(a, "platform_name", return_value="aarch64-n1x"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "meminfo", return_value={"MemTotal": 120 * 2**30}):
  n1x = a.machine_profile()
check("an N1x laptop has a unified NVIDIA GPU of its own family", n1x["ram_class"] == 128 and [(g["vendor"], g["family"], g["memory_gib"]) for g in n1x["gpus"]] == [("nvidia", "nvidia-n1x", 120)])
spark = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "ram_class": 128, "measured": "measurements/model.toml"}])
check("an N1x runs what a GB10 with the same memory runs", a.qualified(spark, n1x, n1x["gpus"][0]) and a.match_class({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128)]}, n1x, n1x["gpus"][0]) is not None)
check("best treats the N1x and GB10 as one class", any("same selectors" in e for e in a.validate_best({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128, recipes=["spark"]), dict(entry, gpu="nvidia-n1x", ram_class=128, recipes=["spark"])]}, {"spark": spark})))
check("GPU peak exceeding capacity does not fit", not a.fits(dict(variant, peak_gpu_gib=49), profile, gpu, "model")[0])
check("RAM fit preserves desktop reserve", not a.fits(dict(variant, peak_ram_gib=57), profile, gpu, "model")[0])
with patch.object(a, "free_memory", return_value=(1, 1)), patch.object(a, "top_memory_users", return_value="browser (12 GiB), editor (4 GiB)"), patch.object(a.time, "sleep"):
  raises("headroom failure lists largest memory users", lambda: a.check_headroom(variant, gpu), "browser (12 GiB), editor (4 GiB)")

payload = b'{"ok":true}'
oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
tree = [{"type": "file", "path": "model.gguf", "size": 2, "lfs": {"oid": "c" * 64, "size": 3}},
        {"type": "file", "path": "tokenizer/x.json", "size": len(payload), "oid": oid}]
with patch.object(a, "hf_tree", return_value=tree):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0]["dest"] = "UD"
  files, roles = a.build_artifacts(changed)
  check("LFS expansion uses published SHA256 and LFS size", files[0].get("sha256") == "c" * 64 and files[0]["size"] == 3)
  check("small files retain git blob identity", files[1].get("git_oid") == oid)
  check("dest prefixes the full repository path", files[1]["dest"] == "UD/tokenizer/x.json" and roles["model"] == "UD/model.gguf")
  changed["artifacts"][0]["model"] = "."
  check("a '.' role names the directory its artifact lands in", a.build_artifacts(changed)[1]["model"] == "UD" and not a.validate_recipe("model", changed, {"server": runtime}))
  del changed["artifacts"][0]["dest"]
  check("a '.' role needs a dest", any("needs a dest" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
  changed["artifacts"][0]["files"] = ["missing*"]
  raises("unmatched artifact pattern raises", lambda: a.build_artifacts(changed), "no file matching")
  changed = copy.deepcopy(recipe)
  changed["artifacts"].append(dict(artifact, repo="other/model"))
  raises("different artifacts cannot overwrite a destination", lambda: a.build_artifacts(changed), "Two artifacts land")
path = Path(os.environ["HOME"]) / "blob"
path.write_bytes(payload)
check("git blob hash verifies intact file", a.verify_file(path, files[1]))
path.write_bytes(payload[:-1] + b"!")
check("git blob hash rejects same-size one-byte corruption", not a.verify_file(path, files[1]))
record = {"recipe": "model", "method": 'say "hello"\nnext\\line', "ctx": 4096,
          "machine": {"platform": "x86", "flags": ["a", "b"]}, "memory": {"peak": 16.5},
          "artifacts": [{"path": "one", "sha256": "a" * 64}, {"path": "two", "git_oid": oid}],
          "short": {"ttft": 1.5, "ok": True}}
check("bench-shaped TOML round-trips tables, arrays and escaped strings", tomllib.loads(bench.toml_dumps(record)) == record)
check("bench names an x86 CPU from cpuinfo", bench.cpu_name("processor\t: 0\nmodel name\t: AMD RYZEN AI MAX+ PRO 395\n", "Model name: ignored") == "AMD RYZEN AI MAX+ PRO 395")
check("bench names an ARM CPU's core types from lscpu", bench.cpu_name("processor\t: 0\nBogoMIPS\t: 2000.00\n", "Vendor ID: ARM\nModel name: Cortex-X925\nModel name: Cortex-A725\nModel name: Cortex-X925\n") == "Cortex-X925 + Cortex-A725")
check("bench reads the NVIDIA driver version", bench.nvidia_driver("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  615.78.08  Release Build  (root@host)\n") == "615.78.08"
      and bench.nvidia_driver("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.95.05  Release Build\n") == "580.95.05" and bench.nvidia_driver("") == "")

# Models run side by side, each on its own port; an agent pointed at one must
# never land on another, so ports are distinct, stable, and freed on removal.
first, second = a.port_for("model-a"), a.port_for("model-b")
check("each model gets its own loopback port", first == a.BASE_PORT and second == a.BASE_PORT + 1)
check("a model keeps its port across lookups", a.port_for("model-a") == first)
ports = a.load_ports()
ports.pop("model-a")
a.write_atomic(a.ports_path(), json.dumps(ports))
check("a removed model's port is reused before a new one", a.port_for("model-c") == first and a.port_for("model-b") == second)

# Materialized file lists cannot escape the store and must carry real hashes.
good = {"dest": "w/model.gguf", "size": 3, "sha256": "a" * 64}
for name, record, roles in (
  ("an escaping destination", {**good, "dest": "../../.bashrc"}, {}),
  ("a malformed hash", {**good, "sha256": "../x"}, {}),
  ("a role naming no file", good, {"model": "other.gguf"}),
  ("a role naming a parent directory", good, {"model": ".."}),
):
  try:
    a.validate_files([record], roles)
  except a.LocalError:
    check(f"a manifest with {name} is refused", True)
  else:
    check(f"a manifest with {name} is refused", False)

a.validate_files([good], {"model": "w"})
check("a manifest role may name the directory holding its files", True)

# A new file set never touches the one a running or previous unit mounts.
def plant(record, body):
  path = a.blob_path(record)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  Path(path).write_bytes(body)
old = {"dest": "w/model.gguf", "size": 3, "sha256": hashlib.sha256(b"old").hexdigest()}
new = {"dest": "w/model.gguf", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
plant(old, b"old")
plant(new, b"new")
first_set = a.materialize("swap", [old], {"model": "w/model.gguf"})
second_set = a.materialize("swap", [new], {"model": "w/model.gguf"})
check("each file set gets its own directory", first_set != second_set)
check("setting up a new file set leaves the previous one intact", Path(a.fileset_dir("swap", first_set), "w/model.gguf").read_bytes() == b"old")

# Exercise typed unit generation, then the real Quadlet parser without starting services.
hostile = '50% $HOME "quoted space" \\tail'
changed = copy.deepcopy(recipe)
changed["server"] = {"args": ["--extra", hostile]}
rt = copy.deepcopy(runtime)
rt["inputs"]["alias"] = {"env": "MODEL_ALIAS"}
rt["health"]["probe"] = ["probe", hostile]
roles = {"model": hostile}
text = a.render_container("model", changed, rt, variant, roles, gpu, "a" * 16)
check("server args escape systemd substitutions, quotes and backslashes", '"50%% $$HOME \\"quoted space\\" \\\\tail"' in text)
check("probe escapes substitutions while retaining JSON argv", "HealthCmd=" + json.dumps(["probe", hostile]).replace("%", "%%").replace("$", "$$") in text)
# Map model through env to pass the same hostile value through Environment=.
env_rt = copy.deepcopy(rt)
env_rt["inputs"]["model"] = {"env": "MODEL_PATH"}
env_text = a.render_container("model", changed, env_rt, variant, roles, gpu, "a" * 16)
check("environment values are quoted and escaped", 'Environment="MODEL_PATH=/models/50%% $$HOME \\"quoted space\\" \\\\tail"' in env_text)
check("units wait for health on Podman 4 too", "Notify=healthy" in text and "PodmanArgs=--sdnotify=healthy" in text)
compat_rt = copy.deepcopy(rt)
compat_rt["run"] = dict(compat_rt.get("run", {}), group_add="keep-groups")
compat_text = a.render_container("model", changed, compat_rt, variant, roles, gpu, "a" * 16)
check("group_add renders for Podman 4's Quadlet", "PodmanArgs=--group-add=keep-groups" in compat_text and "GroupAdd=" not in compat_text)
def fake_platform_run(argv, **kwargs):
  if argv[0] == "omarchy-hw-platform":
    raise FileNotFoundError(argv[0])
  return subprocess.CompletedProcess(argv, 0, "NVIDIA GB10\n", "")
with patch.object(a.subprocess, "run", side_effect=fake_platform_run), patch.object(a.os, "uname", return_value=SimpleNamespace(machine="aarch64")):
  check("without Omarchy a Spark is named from nvidia-smi", a.platform_name() == "aarch64-gb10")
with patch.object(a.subprocess, "run", side_effect=FileNotFoundError), patch.object(a.os, "uname", return_value=SimpleNamespace(machine="x86_64")):
  check("without Omarchy or a Spark the platform stays x86", a.platform_name() == "x86")
smi = "NVIDIA GeForce RTX 3090, 24576, 8.6\nNVIDIA GeForce RTX 3090, 24576, 8.6\n"
found = a.nvidia_gpus(smi)
check("discrete NVIDIA GPUs are listed from nvidia-smi", [g["family"] for g in found] == ["nvidia-rtx3090"] * 2 and found[1]["nvidia_index"] == 1)
check("a discrete GPU's memory is its VRAM, not unified", found[0]["memory_gib"] == 24 and found[0]["unified"] is False and found[0]["compute_cap"] == "8.6")
check("nvidia-smi output that isn't a table lists nothing", a.nvidia_gpus("") == [] and a.nvidia_gpus("No devices were found\n") == [])
check("a workstation card keeps its words in the family", a.nvidia_family("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == "nvidia-rtxpro6000blackwellworkstationedition")
with patch.object(a, "platform_name", return_value="x86"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "nvidia_gpus", return_value=found[:1]), patch.object(a, "meminfo", return_value={"MemTotal": 78 * 2**30}):
  prof = a.machine_profile()
  check("an x86 machine with a 3090 profiles it beside its own RAM", prof["gpus"][0]["family"] == "nvidia-rtx3090" and prof["ram_class"] == 96 and prof["gpus"][0]["vram_class"] == 24)
with patch.object(a, "platform_name", return_value="aarch64-gb10"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "nvidia_gpus", side_effect=AssertionError("a Spark's GPU is unified")), patch.object(a, "meminfo", return_value={"MemTotal": 120 * 2**30}):
  check("a Spark isn't also listed as a discrete GPU", [g["family"] for g in a.machine_profile()["gpus"]] == ["nvidia-gb10"])
with patch.object(a, "nvidia_free_gib", return_value=20.5), patch.object(a, "meminfo", return_value={"MemAvailable": 60 * 2**30}):
  check("free memory on a discrete GPU is its VRAM free", a.free_memory(found[0]) == (60.0, 20.5))
check("VRAM rounds up to its class (23.99 GiB is 24, 200 caps at 96)", a.vram_class(24) == 24 and a.vram_class(23.99) == 24 and a.vram_class(11) == 12 and a.vram_class(200) == 96)
vgpu, rgpu = found[0], {"vendor": "nvidia", "family": "nvidia-gb10", "unified": True, "memory_gib": 120}
vprof = {"platform": "x86", "ram_class": 96, "ram_gib": 78, "gpus": [vgpu]}
check("a discrete GPU is matched on VRAM, a unified one on RAM", a.memory_class(vprof, vgpu) == 24 and a.memory_class(vprof, rgpu) == 96)
vrecipe = {"hardware": [{"gpu": "nvidia-rtx3090", "vram_class": 24}]}
check("a recipe measured at 24 GB VRAM qualifies on a 3090 whatever the RAM", a.qualified(vrecipe, vprof, vgpu) and a.qualified(vrecipe, dict(vprof, ram_class=32), vgpu))
check("a recipe measured at 16 GB VRAM doesn't qualify on a 24 GB card", not a.qualified({"hardware": [{"gpu": "nvidia-rtx3090", "vram_class": 16}]}, vprof, vgpu))
check("a RAM-class entry doesn't qualify a discrete GPU", not a.qualified({"hardware": [{"gpu": "nvidia-rtx3090", "ram_class": 96}]}, vprof, vgpu))
check("a hardware entry names exactly one class", a.class_errors({"gpu": "x", "ram_class": 64, "vram_class": 24}, "w") and a.class_errors({"gpu": "x"}, "w") and not a.class_errors({"gpu": "x", "vram_class": 24}, "w") and a.class_errors({"vram_class": 64}, "w") and a.class_errors({"ram_class": 24}, "w"))
vbest = {"class": [{"gpu": "nvidia-rtx3090", "vram_class": 24, "recipes": ["model"]}]}
check("best.toml matches a discrete GPU on VRAM", a.match_class(vbest, vprof, vgpu) is vbest["class"][0] and a.match_class(vbest, vprof, rgpu) is None)
check("best.toml accepts a vram_class and rejects both", not any("class" in e and "must be" in e for e in a.validate_best(vbest, {"model": {"hardware": [{"gpu": "nvidia-rtx3090", "vram_class": 24}]}})) and a.validate_best({"class": [dict(vbest["class"][0], ram_class=64)]}, {"model": vrecipe}))
with patch.object(a, "free_memory", side_effect=[(60.0, 20.0), (50.0, 5.0)]):
  sampler = a.MemorySampler(found[0])
  sampler.start()
  sampler.sample()
  check("a discrete GPU's peak is its VRAM use, not the RAM peak", sampler.peaks() == (10.0, 15.0))
check("a unified GPU has no counter of its own", not a.gpu_counts_own_memory(rgpu) and a.gpu_counts_own_memory(found[0]) and a.gpu_counts_own_memory({"sysfs": "/sys/x"}))
positional_rt = copy.deepcopy(runtime)
positional_rt["command"] = ["server", "serve"]
positional_rt["inputs"]["model"] = {"positional": True}
_, argv, _, _ = a.server_invocation(positional_rt, recipe, variant, {"model": "w"}, "model")
check("a positional model follows the command", not a.validate_runtime("server", positional_rt) and argv[:3] == ["server", "serve", "/models/w"] and "--model" not in argv)
for location in ("name", "args", "env", "command"):
  r, t = copy.deepcopy(changed), copy.deepcopy(env_rt)
  role_values = dict(roles)
  if location == "name":
    r["name"] = "bad\nname"
  elif location == "args":
    r["server"]["args"] = ["bad\narg"]
  elif location == "env":
    role_values["model"] = "bad\npath"
  else:
    t["command"] = ["bad\ncommand"]
  raises(f"renderer rejects newline in {location}", lambda: a.render_container("model", r, t, variant, role_values, gpu, "a" * 16), "control characters")
bad_probe = copy.deepcopy(rt)
bad_probe["health"]["probe"] = ["bad\nprobe"]
try:
  a.render_container("model", changed, bad_probe, variant, roles, gpu, "a" * 16)
except a.LocalError:
  check("renderer rejects probe newline", True)
else:
  check("renderer rejects probe newline", False)
generator = next((p for p in ("/usr/lib/systemd/user-generators/podman-user-generator", "/usr/lib/systemd/system-generators/podman-system-generator") if Path(p).exists()), None)
if generator:
  quadlets = Path(a.quadlet_dir())
  quadlets.mkdir(parents=True)
  (quadlets / "omarchy-agent-local.container").write_text(text)
  (quadlets / "omarchy-agent-local.network").write_text(a.render_network())
  env = dict(os.environ, QUADLET_UNIT_DIRS=str(quadlets))
  result = subprocess.run([generator, "--user", "--dryrun"], env=env, capture_output=True, text=True)
  check("Quadlet accepts generated units", result.returncode == 0, result.stdout + result.stderr)
  exec_line = next((line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("ExecStart=") and " run " in line), "")
  # systemd doubles literal percent and dollars; shlex handles its emitted quoting here.
  argv = [re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), arg)
          for arg in shlex.split(exec_line.replace("%%", "%").replace("$$", "$"))]
  def option(flag, value):
    return f"{flag}={value}" in argv or any(argv[i:i+2] == [flag, value] for i in range(len(argv)))
  check("Quadlet keeps internal network and container restrictions", option("--network", "omarchy-agent-local") and "--read-only" in argv and option("--cap-drop", "all") and option("--security-opt", "no-new-privileges"), exec_line)
  check("Quadlet publishes only the loopback endpoint", option("--publish", f"127.0.0.1:{a.port_for('model')}:8080"), exec_line)
  check("hostile server argument survives Quadlet as one argument", hostile in argv, repr(argv))
else:
  skip("Quadlet parser unavailable: Podman generator is not installed")

# Paired machines: a runtime spanning two runs on the host network only with a [pair] table,
# binds its API to loopback through flags it reserves, and leaves NCCL settings to the engine.
pair_rt = copy.deepcopy(runtime)
pair_rt.update(command=["server", "serve"], listen={"host": "--host", "port": "--port"}, reserved=["--host", "--port", "API_KEY"],
               run={"devices": ["cdi-nvidia", "infiniband"], "network": "fabric", "memlock": "unlimited"},
               health={"probe": ["curl", "http://127.0.0.1:@PORT@/health"]},
               pair={"nodes": 2, "master_port": 29620, "state": "/state",
                     "env": {"NCCL_COMM_ID": "@MASTER_ADDR@:@MASTER_PORT@", "TF_RANK": "@RANK@"}})
check("a paired runtime fixture is accepted", not a.validate_runtime("pair", pair_rt), "\n".join(a.validate_runtime("pair", pair_rt)))
for name, mutate, fragment in (
  ("host network without a pair", lambda r: r.pop("pair"), "only for runtimes with a [pair]"),
  ("a pair on the internal network", lambda r: r["run"].pop("network"), "needs run.network"),
  ("a pair without bind flags", lambda r: r.pop("listen"), ".listen"),
  ("bind flags in the command", lambda r: r.update(command=["server", "--port", "1"]), "sets --host, --port itself"),
  ("unreserved bind flags", lambda r: r.update(reserved=["API_KEY"]), "must reserve"),
  ("NCCL settings in the runtime", lambda r: r["pair"]["env"].update(NCCL_IB_HCA="mlx5_0"), "NCCL_"),
  ("three nodes", lambda r: r["pair"].update(nodes=3), "only 2"),
  ("no state path", lambda r: r["pair"].pop("state"), "pair.state"),
  ("a state path over the models", lambda r: r["pair"].update(state="/models"), "pair.state"),
  ("the link's NCCL interface", lambda r: r["pair"]["env"].update(NCCL_SOCKET_IFNAME="eth0"), "NCCL_SOCKET_IFNAME"),
):
  changed = copy.deepcopy(pair_rt)
  mutate(changed)
  check(f"paired runtime rejects {name}", any(fragment in e for e in a.validate_runtime("pair", changed)), "\n".join(a.validate_runtime("pair", changed)))

spark = {"vendor": "nvidia", "family": "nvidia-gb10", "memory_gib": 121.9, "unified": True, "sysfs": ""}
pair_gpu = dict(spark, family="nvidia-gb10-pair", nodes=2, memory_gib=121.7, peer="10.99.0.2")
pair_profile = {"platform": "aarch64-gb10", "ram_class": 128, "ram_gib": 121.9, "gpus": [spark, pair_gpu]}
check("a pair outranks its own node", a.choose_gpu(pair_profile)["family"] == "nvidia-gb10-pair")
single = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "ram_class": 128, "measured": "m.toml"}], runtime="server")
paired = dict(recipe, hardware=[{"gpu": "nvidia-gb10-pair", "ram_class": 128, "measured": "m.toml"}], runtime="pair",
              variant=[{"ctx": 4096, "peak_ram_gib": 104, "peak_gpu_gib": 104}])
check("a pair machine runs single-node recipes on its own GPU", a.recipe_gpu(pair_profile, single)["family"] == "nvidia-gb10" and a.recipe_gpu(pair_profile, paired)["family"] == "nvidia-gb10-pair")
nv_rt = dict(runtime, run={"devices": ["cdi-nvidia"]})
pair_catalog = SimpleNamespace(recipes={"single": single, "paired": paired}, runtimes={"server": nv_rt, "pair": pair_rt},
                               best={"class": [{"gpu": "nvidia-gb10", "ram_class": 128, "recipes": ["single"]},
                                               {"gpu": "nvidia-gb10-pair", "ram_class": 128, "recipes": ["paired"]}]})
check("best on a pair is the pair's recipe", a.resolve_best(pair_catalog, pair_profile)[0] == "paired")
check("best without the pair is the node's own", a.resolve_best(pair_catalog, dict(pair_profile, gpus=[spark]))[0] == "single")
small = dict(pair_profile, gpus=[spark, dict(pair_gpu, memory_gib=100)])
check("best falls back to the node's class when the pair recipe doesn't fit", a.resolve_best(pair_catalog, small)[0] == "single")
check("a paired runtime never runs on one node, nor a single one on the pair", not a.runtime_supports(pair_rt, spark) and not a.runtime_supports(nv_rt, pair_gpu))
pair_text = a.render_container("paired", paired, pair_rt, paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16)
pair_port = a.port_for("paired")
check("the pair unit shares the host network and publishes nothing", "Network=host" in pair_text and "PublishPort" not in pair_text)
check("the pair unit binds its API to loopback on the model's port", f'"--host" "127.0.0.1" "--port" "{pair_port}"' in pair_text and f"127.0.0.1:{pair_port}/health" in pair_text)
check("the pair unit mounts rank 0's state folder writable and reads the link from its environment file", f"models/paired/{'b' * 16}.rank0:/state\n" in pair_text and "EnvironmentFile=%h/.local/state/omarchy/agent-local/pair/paired.env" in pair_text)
check("the pair unit starts and stops rank 1 with itself", f"ExecStartPre={a.hook_command()} link-worker start paired" in pair_text and f"ExecStopPost=-{a.hook_command()} link-worker stop paired" in pair_text)
_, rank_argv, _, _ = a.head_argv(pair_rt, paired, paired["variant"][0], {"model": "model"}, "paired")
check("both ranks run the runtime's command, bound to loopback", rank_argv[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))] and rank_argv[:2] == ["server", "serve"])
# Each rank gets its own link settings, and rank 1 the head's command inside the same restrictions.
setup = {"schema": 1, "recipe_id": "paired", "runtime": pair_rt, "recipe": paired, "variant": paired["variant"][0], "roles": {"model": "model"},
         "fileset": "e" * 16, "pair": {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64}}
a.write_pair_setup(setup)
here = {"addr": "10.99.1.1", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "5"}
there = {"addr": "10.99.1.2", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "3"}
with patch.object(a, "pair_link", return_value=(here, there)), patch.object(a, "peer_headroom") as headroom, patch.object(a, "pair_ssh") as ssh, patch.object(a, "pair_limits") as limits:
  a.pair_worker_start("paired")
head_env = dict(line.split("=", 1) for line in Path(a.pair_env_path("paired")).read_text().splitlines())
worker = shlex.split(ssh.call_args.args[1])
worker_env = dict(worker[i + 1].split("=", 1) for i, word in enumerate(worker) if word == "-e")
check("rank 0's environment is rank 0's side of the link", head_env == {"NCCL_COMM_ID": "10.99.1.1:29620", "TF_RANK": "0", "NCCL_SOCKET_IFNAME": "enp1", "NCCL_IB_HCA": "rocep1,roceP2", "NCCL_IB_GID_INDEX": "5"}, repr(head_env))
check("rank 1's environment is rank 1's side, same rendezvous", worker_env["TF_RANK"] == "1" and worker_env["NCCL_COMM_ID"] == "10.99.1.1:29620" and worker_env["NCCL_IB_GID_INDEX"] == "3" and "API_KEY" not in worker_env, repr(worker_env))
check("rank 1 starts on the peer as its user, with the head's command and the same restrictions",
      ssh.call_args.args[0] == "bob@10.99.1.2" and headroom.called
      and limits.called and all(flag in worker for flag in ("--read-only", "no-new-privileges", "/dev/infiniband", "f" * 64))
      and worker[worker.index("--cap-drop") + 1] == "all"
      and f"/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}:/models:ro" in worker
      and f"/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}.rank1:/state" in worker
      and worker[worker.index("f" * 64) + 1:][:2] == ["server", "serve"] and worker[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))], repr(worker))
os.unlink(a.pair_setup_path("paired"))
# Setup copies the image (only when the peer lacks it) and the file set, and makes both ranks' state folders.
ran = []
def fake_run(argv, **kwargs):
  ran.append(argv)
  return SimpleNamespace(returncode=0, stdout="f" * 64 + "\n", stderr="")
manifest = {"fileset": "e" * 16}
with patch.object(a.subprocess, "run", side_effect=fake_run), patch.object(a.subprocess, "Popen") as save, \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as ssh, patch.object(a, "load_installing", return_value=None):
  a.pair_setup("paired", manifest, pair_rt, {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob"})
rsync = next(argv for argv in ran if argv[0] == "rsync")
check("setup records the peer and the image by ID", manifest["pair"] == {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64})
check("setup copies the file set to the peer, not an image it already has", not save.called and rsync[-1] == f"bob@10.99.1.2:/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}/")
check("setup makes both ranks' state folders", Path(a.rank_dir("paired", "e" * 16, 0)).is_dir() and any(f"{'e' * 16}.rank1" in call.args[1] for call in ssh.call_args_list))
check("the pair unit keeps the container restrictions", all(line in pair_text for line in ("ReadOnly=true", "DropCapability=all", "NoNewPrivileges=true", "AddDevice=/dev/infiniband")))
if generator:
  quadlets = Path(a.quadlet_dir())
  (quadlets / "omarchy-agent-local.container").write_text(pair_text)
  result = subprocess.run([generator, "--user", "--dryrun"], env=dict(os.environ, QUADLET_UNIT_DIRS=str(quadlets)), capture_output=True, text=True)
  check("Quadlet accepts the pair unit", result.returncode == 0 and "--network host" in result.stdout.replace("=", " "), result.stdout[-2000:] + result.stderr)
scope = {}
exec(a.FABRIC_SIDE, scope)
check("the fabric probe runs on its own and finds no RoCE device on loopback", (scope["fabric_side"]("127.0.0.1") or {}).get("hca") is None)
base = Path(a.data_dir(), "models", "keep")
for name in ("c" * 16, "c" * 16 + ".rank0", "d" * 16, "d" * 16 + ".rank0"):
  (base / name).mkdir(parents=True)
a.prune_filesets("keep", "c" * 16)
check("pruning keeps a file set's rank inputs and drops the others'", sorted(p.name for p in base.iterdir()) == ["c" * 16, "c" * 16 + ".rank0"])
with patch.object(a, "platform_name", return_value="aarch64-gb10"), patch.object(a, "amd_gpus", side_effect=lambda: []):
  check("no pair without a record of one", "pair" not in a.machine_profile())
  Path(a.pair_path()).parent.mkdir(parents=True, exist_ok=True)
  Path(a.pair_path()).write_text(json.dumps({"peer": "10.99.0.2", "host": "spark2", "home": "/home/u", "ram_gib": 121.9}))
  cached = a.machine_profile()
  check("a recorded pair adds the pair GPU without probing", [g["family"] for g in cached["gpus"]] == ["nvidia-gb10", "nvidia-gb10-pair"] and cached["gpus"][1]["peer"] == "10.99.0.2", json.dumps(cached))
  os.unlink(a.pair_path())
  # The record keeps the SSH user link setup was given, and routes by the address alone.
  routed = []
  with patch.object(a, "pair_python", return_value={"gpu": "NVIDIA GB10", "ram_total": 128 * 2**30, "home": "/home/bob", "host": "spark2"}), \
       patch.object(a, "_fabric", side_effect=lambda address: routed.append(address) or {"hca": "rocep1s0f0"}):
    a.record_pair(pair_profile, "bob@10.99.1.2")
  check("a recorded peer keeps the SSH user it was set up with", a.cached_pair(pair_profile)["peer"] == "bob@10.99.1.2" and routed == ["10.99.1.2"], repr(routed))
  with patch.object(a, "local_ports", return_value={"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}), \
       patch.object(a, "pair_python", return_value={"enp1": ["10.99.1.2/24"]}):
    check("link setup reaches the peer at its own cable address, as the same user", a.cable_peer("bob@spark-760a.local") == "bob@10.99.1.2")
  with patch.object(a, "pair_python", return_value={"gpu": "", "ram_total": 128 * 2**30, "home": "/", "host": "x"}):
    raises("only a GB10 is recorded as a peer", lambda: a.record_pair(pair_profile, "10.99.1.9"), "isn't a DGX Spark")
  os.unlink(a.pair_path())

for name, script in (("host", a.HOST_SETUP.replace("@CDI_COMPAT@", a.CDI_COMPAT)), ("link", a.LINK_SETUP), ("firewall", a.FIREWALL_SETUP)):
  result = subprocess.run(["bash", "-n", "-c", script], capture_output=True, text=True)
  check(f"link setup's {name} script parses", result.returncode == 0, result.stderr)
# Cable addressing finishes whatever an earlier setup left, and leaves ports that aren't shared alone.
fresh = a.plan_link({"enp1": [], "enP2": []}, {"enp1": [], "enP2": []})
check("a fresh cable gets a /24 per port, .1 here and .2 there", fresh == (["enP2=10.99.0.1/24", "enp1=10.99.1.1/24"], ["enP2=10.99.0.2/24", "enp1=10.99.1.2/24"]), repr(fresh))
half = a.plan_link({"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}, {"enp1": []})
check("an interrupted setup gives the peer the other end of this side's /24 and ignores unshared ports", half == ([], ["enp1=10.99.1.2/24"]), repr(half))
check("a cable addressed on both ends is left alone", a.plan_link({"enp1": ["10.99.1.1/24"]}, {"enp1": ["10.99.1.2/24"]}) == ([], []))
check("a new /24 avoids networks either end already uses", a.plan_link({"enp1": [], "enp2": ["10.99.0.1/24"]}, {"enp1": [], "enp2": ["10.99.0.2/24"]})[0] == ["enp1=10.99.1.1/24"])
rules = a.link_rules({"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}, {"enp1": ["10.99.1.2/24"], "lan0": ["192.168.1.9/24"], "ib9": ["10.0.0.2/8"]}, {"enp1"})
check("the firewall lets in only the other end's address, on the rails alone", rules == (["enp1=10.99.1.2"], ["enp1=10.99.1.1"]), repr(rules))
with patch.object(a, "pair_ssh", side_effect=a.LocalError("unreachable")), patch.object(a, "known_models", return_value=[]):
  raises("removing a paired model stops while its peer can't be reached", lambda: a.pair_remove({"recipe_id": "paired", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u", "image_id": "c" * 64}}), "--local-only")
  a.pair_remove({"recipe_id": "paired", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u", "image_id": "c" * 64}}, local_only=True)
  check("--local-only removes a paired model whose peer is gone", True)
compile(a.CDI_COMPAT, "nvidia-cdi-podman-compat", "exec")
check("the CDI compatibility script compiles", True)

# A Spark on the cable before setup is offered by the address mDNS names; a linked one by its host.
with patch.object(a, "cable_neighbors", return_value=["fe80::1%enp1", "fe80::2%enp1"]), patch.object(a, "mdns_names", return_value={"fe80::2": "spark-760a"}):
  offered = a.link_probe(pair_profile)
check("a Spark on the cable waiting for setup is offered by its mDNS name", offered == {"state": "available", "address": "fe80::2%enp1", "host": "spark-760a"} and a.link_state() == offered, repr(offered))
check("a linked Spark is shown as linked", a.link_probe(dict(pair_profile, pair={"host": "spark2"})) == {"state": "linked", "host": "spark2"})
with patch.object(a, "cable_neighbors", return_value=[]):
  check("an empty cable offers nothing", a.link_probe(pair_profile) == {"state": "none"})
check("machines that can't pair offer nothing", a.link_probe(dict(pair_profile, platform="x86")) == {"state": "none"})
with patch.object(a, "cable_neighbors", return_value=["fe80::2%enp1"]), patch.object(a, "mdns_names", return_value={}), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
  a.link_notify(pair_profile)
  a.link_notify(pair_profile)
notices = [call.args[0] for call in sent.call_args_list if call.args and call.args[0][:1] == ["omarchy-notification-send"]]
check("a newly cabled Spark is announced once, with link setup as its action", len(notices) == 1 and notices[0][-1] == "omarchy-agent-local link setup", repr(notices))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
# A refresh that found the Spark first doesn't stop the announcement.
with patch.object(a, "cable_neighbors", return_value=["fe80::3%enp1"]), patch.object(a, "mdns_names", return_value={}):
  a.link_probe(pair_profile)
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
    a.link_notify(pair_profile)
check("a Spark a refresh found first is still announced", any(c.args and c.args[0][:1] == ["omarchy-notification-send"] for c in sent.call_args_list))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
# A send that failed (no notification daemon yet, say) is tried again next time.
with patch.object(a, "cable_neighbors", return_value=["fe80::4%enp1"]), patch.object(a, "mdns_names", return_value={}):
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="")):
    a.link_notify(pair_profile)
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
    a.link_notify(pair_profile)
check("a failed announcement is tried again", any(c.args and c.args[0][:1] == ["omarchy-notification-send"] for c in sent.call_args_list))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
os.unlink(a.link_path())
# The parsers read what the tools print.
ping = """PING ff02::1%enp1 (ff02::1%enp1) 56 data bytes
64 bytes from fe80::adfd:1ade:3de8:d49b%enp1: icmp_seq=1 ttl=64 time=0.067 ms
64 bytes from fe80::aba5:f382:c378:65aa%enp1: icmp_seq=1 ttl=64 time=0.133 ms
64 bytes from fe80::5a6b:10b2:a79:3111%enp1: icmp_seq=1 ttl=64 time=0.472 ms
64 bytes from fe80::5a6b:10b2:a79:3111%enp1: icmp_seq=2 ttl=64 time=0.401 ms (DUP!)
"""
own = a.own_link_local(json.dumps([{"ifname": "enp1", "addr_info": [{"family": "inet6", "local": "fe80::adfd:1ade:3de8:d49b", "scope": "link"}]},
                                   {"ifname": "enp2", "addr_info": [{"family": "inet6", "local": "fe80::aba5:f382:c378:65aa", "scope": "link"},
                                                                    {"family": "inet6", "local": "fd00::1", "scope": "global"}]}]))
check("own link-local addresses are read from ip -j", own == {"fe80::adfd:1ade:3de8:d49b", "fe80::aba5:f382:c378:65aa"}, repr(own))
check("a ping's answers drop this machine's own addresses and repeats", a.ping_answers(ping, own) == ["fe80::5a6b:10b2:a79:3111"])
check("a failed ping or ip read gives nothing", a.ping_answers("", own) == [] and a.own_link_local("not json") == set())
avahi = "+;enp1;IPv6;spark-760a;SSH Remote Terminal;local\n=;enp1;IPv6;spark-760a;SSH Remote Terminal;local;spark-760a.local;fe80::9514:4be5:80e5:7b8b;22;\n=;enP7s7;IPv4;spark-760a;SSH Remote Terminal;local;spark-760a.local;192.168.1.162;22;\n"
check("mDNS names come from avahi-browse's resolved IPv6 lines", a.mdns_ssh_hosts(avahi) == {"fe80::9514:4be5:80e5:7b8b": "spark-760a"})
# Once a pair is recorded, status says linked whatever the last probe saw.
a.write_atomic(a.link_path(), json.dumps({"state": "available", "address": "fe80::2%enp1", "host": "x"}))
with patch.object(a, "known_models", return_value=[]), patch.object(a, "installing_models", return_value=[]), patch.object(a, "resolve_best", side_effect=a.LocalError("none")):
  linked = a.status_record(SimpleNamespace(version="test", best={}, recipes={}), dict(pair_profile, pair={"host": "spark2"}))["link"]
check("status shows a recorded pair as linked, not a stale offer", linked == {"state": "linked", "host": "spark2"}, repr(linked))
os.unlink(a.link_path())
check("the cable hook parses", subprocess.run(["bash", "-n", os.path.join(os.environ["ENGINE_ROOT"], "system/90-omarchy-local-ai-link")]).returncode == 0)

# A model that stopped unexpectedly says why: this start's own recorded reason, else the
# server's last error line, else systemd's result.
journal = "info: weights read\nerror: ncclCommInitRank: unhandled system error (run with NCCL_DEBUG=INFO for details) (2)\nsome trailing line"
started = int(__import__("time").time() - 60)
with patch.object(a, "unit_props", return_value={"InactiveExitTimestamp": "@%d" % started}), patch.object(a, "journal_tail", return_value=journal) as tail:
  a.clear_error("crashy")
  check("a crash shows the server's own last error", a.failure_reason("crashy", "exit-code").startswith("ncclCommInitRank: unhandled system error"))
  check("only this start's log is searched, not an earlier run's", tail.call_args.args[2] == started * 1000, repr(tail.call_args))
  a.record_error("crashy", "Locked memory limited: run link setup")
  check("a reason the start already recorded isn't overwritten", a.failure_reason("crashy", "exit-code") == "Locked memory limited: run link setup")
with patch.object(a, "unit_props", return_value={"InactiveExitTimestamp": "@%d" % int(__import__("time").time() + 60)}), patch.object(a, "journal_tail", return_value="nothing useful"):
  check("an older reason gives way, and with no server error systemd's result is shown", a.failure_reason("crashy", "exit-code") == "Stopped unexpectedly (exit-code)")
a.clear_error("crashy")
# logs prints the last error, the model's journal and, for a pair, rank 1's log from the peer.
a.record_error("crashy", "Locked memory limited: run link setup")
a.write_atomic(a.model_manifest_path("crashy"), json.dumps({"recipe_id": "crashy", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u"}}))
printed = []
with patch.object(a, "say", side_effect=printed.append), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Nov 10 rank0 line", stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="rank1 line", stderr="")) as peer_logs:
  a.cmd_logs(["crashy", "-n", "50"])
check("logs shows the last error, this machine's log and rank 1's from the peer",
      printed[0] == "Last error: Locked memory limited: run link setup" and "Nov 10 rank0 line" in printed and "rank1 line" in printed
      and "podman logs --tail 50" in peer_logs.call_args.args[1], repr(printed))
os.unlink(a.model_manifest_path("crashy"))
a.clear_error("crashy")
# A failed first setup's reason lives in its installing record.
a.write_atomic(a.installing_path("crashy"), json.dumps({"recipe_id": "crashy", "phase": "failed", "error": "Download failed",
                                                       "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u"}}))
printed = []
with patch.object(a, "say", side_effect=printed.append), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="rank1 failed", stderr="")):
  a.cmd_logs(["crashy"])
check("logs shows a failed setup's saved reason", printed[:1] == ["Last error: Download failed"], repr(printed))
check("a failed first paired start still shows rank 1's log", "rank1 failed" in printed, repr(printed))
os.unlink(a.installing_path("crashy"))

# The default root must work without an Omarchy checkout or a catalog override.
check("installed catalog loads and validates from the engine directory", not a.load_catalog().errors() and a.load_catalog().root == os.environ["ENGINE_ROOT"])
root = Path(os.environ["HOME"])
repo = root / "catalog-fixture"
repo.mkdir()
def write_toml(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(bench.toml_dumps(value))
write_toml(repo / "catalog.toml", {"schema": 1})
write_toml(repo / "best.toml", {"class": [entry]})
write_toml(repo / "runtimes/server.toml", runtime)
write_toml(repo / "recipes/model.toml", recipe)
write_toml(repo / "reviews/server.toml", {"image": runtime["image"], "publisher": "Fixture", "reviewer": "Tester", "date": "2026-01-01", "license_url": "https://example.test/license", "license_sha256": "a" * 64, "checked": ["key", "network"]})
write_toml(repo / "measurements/model.toml", {"ctx": 4096})
os.environ["OMARCHY_LOCAL_AI_CATALOG"] = str(repo)
check("catalog root can be overridden", a.load_catalog().root == str(repo))
write_toml(repo / "catalog.toml", {"schema": 999})
raises("invalid installed catalog is rejected", a.load_catalog, "failed validation")
write_toml(repo / "catalog.toml", {"schema": 1})
# The package command follows /usr/bin-style symlinks to find its engine.
link = root / "bin/omarchy-local-ai"
link.parent.mkdir()
link.symlink_to(Path(os.environ["ENGINE_ROOT"]) / "omarchy-local-ai")
result = subprocess.run([str(link), "validate", os.environ["ENGINE_ROOT"]], cwd=root, capture_output=True, text=True)
check("entry script imports its engine through a symlink", result.returncode == 0, result.stdout + result.stderr)
with patch.object(a, "machine_profile", return_value=profile), patch.object(a, "running_models", return_value=set()), patch.object(a, "known_models", return_value=[]):
  check("quiet refresh succeeds", a.cmd_refresh(["--quiet"]) == 0)
check("refresh writes both panel caches", Path(a.state_dir(), "list.json").is_file() and Path(a.state_dir(), "status.json").is_file())

# Removing a recipe drops the blobs only it used, even one still linked from
# the seed copy it was set up from, and keeps those another recipe links.
store = Path(a.data_dir())
for name in ("kept", "seeded"):
  (store / "blobs/sha256").mkdir(parents=True, exist_ok=True)
  (store / "blobs/sha256" / name).write_text(name)
for rid, name in (("gone", "kept"), ("gone", "seeded"), ("other", "kept")):
  (store / "models" / rid).mkdir(parents=True, exist_ok=True)
  os.link(store / "blobs/sha256" / name, store / "models" / rid / name)
seed = Path(os.environ["HOME"]) / "seed"
seed.mkdir()
os.link(store / "blobs/sha256/seeded", seed / "seeded")
with patch.object(a, "write_unit_files"):
  a._remove_files("gone")
check("removal drops a blob only a seed copy still links", not (store / "blobs/sha256/seeded").exists() and (seed / "seeded").read_text() == "seeded")
check("removal keeps a blob another recipe links", (store / "blobs/sha256/kept").exists())

# An interrupted removal leaves the model listed, so remove can finish it.
Path(a.model_manifest_path("halfway")).parent.mkdir(parents=True, exist_ok=True)
Path(a.model_manifest_path("halfway")).write_text(json.dumps({"files": []}))
with patch.object(a, "unit_active", return_value=False), patch.object(a, "write_status"), patch.object(a, "_remove_files", side_effect=KeyboardInterrupt):
  try:
    a.cmd_remove(["halfway"])
  except KeyboardInterrupt:
    pass
check("an interrupted removal keeps the model's manifest", Path(a.model_manifest_path("halfway")).exists() and not Path(a.removing_path("halfway")).exists())

# Validation accepts exactly the run-option shapes the generator implements.
for option, value in (("devices", "render"), ("devices", {"render": True}),
                      ("group_add", ["keep-groups"]), ("memlock", ["unlimited"])):
  changed = dict(runtime, run={option: value})
  check(f"run option {option} rejects {value!r}", bool(a.validate_runtime("server", changed)))
check("supported run-option shapes validate", not a.validate_runtime("server", dict(runtime, run={"devices": ["render", "kfd"], "group_add": "keep-groups", "memlock": "unlimited"})))
for fileset in (None, "", "../outside", "a" * 16 + "\n"):
  raises("renderer requires a safe file set", lambda: a.render_container("model", recipe, runtime, variant, {"model": "model.gguf"}, gpu, fileset), "file set")
check("unit mounts the exact immutable file set", "models/model/" + "a" * 16 + ":/models:ro" in text)
check("unit injects key through Podman secrets", f"Secret={a.SECRET},type=env,target={a.KEY_ENV}" in text and f"Secret={a.SECRET},type=env,target=API_KEY" in text)

# A saved setup is installed even if it has never started. Replacing it only
# changes the current manifest, and removal considers surviving manifests.
manifest = {"schema": 1, "recipe_id": "current", "recipe": recipe, "runtime": runtime,
            "runtime_id": "server", "image": runtime["image"], "variant": variant,
            "files": [], "roles": {"model": "model.gguf"}, "fileset": "a" * 16,
            "gpu": gpu["family"], "profile": profile,
            "observed": {"load_seconds": 12, "ram_used_gib": 16, "gpu_used_gib": 16}}
path = a.save_model(manifest, started=False)
check("a current manifest alone means installed", a.recipe_installed("current"))
updated = dict(manifest, variant=dict(variant, ctx=8192))
check("saving replaces the same current manifest", a.save_model(updated, started=False) == path and a.load_model("current")["variant"]["ctx"] == 8192)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  pending = a.write_status(a.load_catalog(), profile, {"current": "activating"}, pending={"current": dict(updated, observed=None)})
check("pending activation uses the current manifest's observations", pending["models"]["current"]["observed"] == manifest["observed"] and pending["models"]["current"]["unit"] == "activating")
a.save_model(dict(manifest, recipe_id="survivor"), started=False)
with patch.object(a, "known_models", return_value=["current", "survivor"]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run") as rmi:
  a._remove_files("current")
  check("removal retains an image used by a surviving current manifest", not rmi.called)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as rmi:
  a._remove_files("current")
  check("removal drops an image no surviving manifest uses", rmi.call_args.args[0] == ["podman", "rmi", runtime["image"]])

# A recipe that takes images says so to the agents, and only text and image are allowed.
image_recipe = {**copy.deepcopy(recipe), "input": ["text", "image"]}
check("a recipe can take images", not [e for e in a.validate_recipe("model", image_recipe, {"server": runtime}, root) if ".input" in e])
check("a recipe has to take text", any(".input" in e for e in a.validate_recipe("model", {**copy.deepcopy(recipe), "input": ["image"]}, {"server": runtime}, root)))
with patch.object(a, "resolve_model", return_value={"recipe_id": "model", "recipe": image_recipe, "variant": {"ctx": 4096}}):
  overlay = json.loads(open(a.opencode_overlay("model")[0]).read())["provider"][a.OVERLAY_PROVIDER]["models"]["model"]
  check("the OpenCode overlay declares image input", overlay["attachment"] and overlay["modalities"]["input"] == ["text", "image"])

# pi's own extension lists every model set up, each at its own port, and
# follows them: none without pi's folder, gone with the last model.
def pi_registered(path):
  text = open(path).read()
  name, _, rest = text.split("pi.registerProvider(", 1)[1].partition(", ")
  return json.loads(name), json.loads(rest.rsplit(");", 1)[0])
pi_set_up = {"alpha": {"recipe_id": "alpha", "recipe": image_recipe, "variant": {"ctx": 4096}},
             "beta": {"recipe_id": "beta", "recipe": {**copy.deepcopy(recipe), "name": "Beta"}, "variant": {"ctx": 8192}}}
pi_ext = Path(a.pi_extension_path())
with patch.object(a, "known_models", return_value=["alpha", "beta"]), patch.object(a, "load_model", side_effect=pi_set_up.get):
  a.sync_pi_extension()
  check("no pi folder, no pi extension", not pi_ext.exists() and not pi_ext.parent.parent.exists())
  pi_ext.parent.parent.mkdir(parents=True)
  a.sync_pi_extension()
  name, provider = pi_registered(pi_ext)
  urls = {m["id"]: m["baseUrl"] for m in provider["models"]}
  check("pi's extension registers the omarchy-local provider", name == "omarchy-local" == a.OVERLAY_PROVIDER)
  check("pi's extension lists every model set up at its own port", urls == {"alpha": a.endpoint("alpha"), "beta": a.endpoint("beta")} and urls["alpha"] != urls["beta"], repr(urls))
  check("pi's extension reads the key from its file", provider["apiKey"] == "!cat -- " + shlex.quote(a.key_path()))
  check("pi's extension carries each model's context and input", [(m["contextWindow"], m["input"]) for m in provider["models"]] == [(4096, ["text", "image"]), (8192, ["text"])])
  with patch.object(a, "resolve_model", return_value=pi_set_up["beta"]):
    path, overlay_name, overlay_model = a.pi_overlay("beta")
  name, provider = pi_registered(path)
  check("a local launch overlay is the same provider, the launched model first and the rest after", (name, overlay_name, overlay_model) == ("omarchy-local", "omarchy-local", "beta") and [m["id"] for m in provider["models"]] == ["beta", "alpha"])
  mtime = pi_ext.stat().st_mtime_ns
  a.sync_pi_extension()
  check("an unchanged list leaves pi's extension alone", pi_ext.stat().st_mtime_ns == mtime)
with patch.object(a, "known_models", return_value=[]):
  a.sync_pi_extension()
  check("pi's extension goes with the last model", not pi_ext.exists())
import fcntl
def sync_while_locked():
  # Another writer can't take the status lock while pi's extension is written.
  with open(os.path.join(a.state_dir(), "status.lock"), "w") as probe:
    try:
      fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
      raise AssertionError("pi's extension was written outside the status lock")
    except BlockingIOError:
      pass
  raise OSError("read-only")
with patch.object(a, "status_record", return_value={"models": {}, "share_error": ""}), patch.object(a, "publish_shares"), patch.object(a, "sync_pi_extension", side_effect=sync_while_locked) as sync:
  check("a status write syncs pi's extension under its lock and survives it failing", "models" in a.write_status(a.load_catalog(), profile) and sync.called)
check("agents default to a 16k reply cap", a.max_output_tokens({"recipe": recipe, "variant": {"ctx": 131072}}) == 16384)
capped = {**copy.deepcopy(recipe), "server": {"args": ["--max-tokens", "32768", "--vision"]}}
check("agents get the server's reply cap", a.max_output_tokens({"recipe": capped, "variant": {"ctx": 131072}}) == 32768)
check("a variant's reply cap counts too", a.max_output_tokens({"recipe": recipe, "variant": {"ctx": 131072, "args": ["--max-tokens=65536"]}}) == 65536)
shipped = tomllib.load(open(os.path.join(os.environ["ENGINE_ROOT"], "recipes", "qwen3.8-27b-tensorfold.toml"), "rb"))
check("shipped recipes give agents a 64k reply cap", all(a.max_output_tokens({"recipe": shipped, "variant": v}) == 65536 for v in shipped["variant"]))
# Agents re-send every screenshot they have read, and tensorfold takes only 4 images per request by default.
recipe_dir = os.path.join(os.environ["ENGINE_ROOT"], "recipes")
seeing = [tomllib.load(open(os.path.join(recipe_dir, name), "rb"))["server"]["args"] for name in sorted(os.listdir(recipe_dir)) if "tensorfold" in name]
seeing = [args for args in seeing if "--vision" in args]
check("tensorfold recipes that take images allow 50 per request", seeing and all("--vision-max-images" in args and args[args.index("--vision-max-images") + 1] == "50" for args in seeing))
check("the reply cap never exceeds the context", a.max_output_tokens({"recipe": capped, "variant": {"ctx": 8192}}) == 8192)
with patch.object(a, "resolve_model", return_value={"recipe_id": "model", "recipe": capped, "variant": {"ctx": 131072}}):
  check("the pi overlay carries the server's reply cap", '"maxTokens": 32768' in open(a.pi_overlay("model")[0]).read())
  check("the OpenCode overlay carries the server's reply cap", json.loads(open(a.opencode_overlay("model")[0]).read())["provider"][a.OVERLAY_PROVIDER]["models"]["model"]["limit"]["output"] == 32768)

# Setups saved while memory is busy still owe authentication/model-ID checks.
cat = a.load_catalog()
rid = next(iter(cat.recipes))
rec = cat.recipes[rid]
rt = cat.runtimes[rec["runtime"]]
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "preflight"), patch.object(a, "build_artifacts", return_value=([], {"model": "model.gguf"})), patch.object(a, "materialize", return_value="a" * 16), patch.object(a, "ensure_image"), patch.object(a, "ensure_key", return_value="test-key"), patch.object(a, "unit_active", return_value=False), patch.object(a, "check_headroom", side_effect=a.LocalError(a.NOT_ENOUGH_MEMORY)), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"), patch.object(a, "prune_filesets"):
  check("memory-busy first setup succeeds without activation", a.cmd_use([rid]) == 0 and a.load_model(rid)["unchecked"])
for response, succeeds in (([(200, b"")], False),
                           ([(401, b""), (200, b'{"data":[{"id":"other"}]}')], False),
                           ([(401, b""), (200, json.dumps({"data": [{"id": rid}]}).encode())], True)):
  with patch.object(a, "unit_active", return_value=False), patch.object(a, "write_status"), patch.object(a, "evict_idle_weights"), patch.object(a, "check_headroom", return_value=(100.0, 90.0)), patch.object(a, "free_memory", return_value=(52.0, 64.0)), patch.object(a, "ensure_key", return_value="test-key"), patch.object(a, "http_status", side_effect=response) as http, patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as systemctl:
    if succeeds:
      a._start(SimpleNamespace(recipe=rid), cat, profile)
      check("first checked start clears the unchecked flag", not a.load_model(rid).get("unchecked") and http.call_count == 2)
      check("a start records the memory the model took", a.load_model(rid)["observed"]["ram_used_gib"] == 48.0)
    else:
      raises("first start refuses failed authentication or wrong model ID", lambda: a._start(SimpleNamespace(recipe=rid), cat, profile))
      check("failed first check stops the model and retains its check obligation", ("stop", a.unit_name(rid)) in [call.args for call in systemctl.call_args_list] and a.load_model(rid)["unchecked"])
check("a pair's memory is both ranks'", a.ram_used_gib({"pair": {"peer": "10.99.1.2"}}, 100.0, 10.0, 120.0, 29.0) == 181.0 and a.ram_used_gib({"pair": {"peer": "10.99.1.2"}}, 100.0, 10.0, None, 29.0) == 180.0 and a.ram_used_gib({}, 100.0, 10.0, None, None) == 90.0)
check("a pair's memory is shown against both machines", a.nodes_ram_gib({"pair": {"peer": "10.99.1.2"}}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}) == 243.6 and a.nodes_ram_gib({}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}) == 121.9)
check("a pair's memory total is its own peer's, not a later link's", a.nodes_ram_gib({"pair": {"peer": "10.99.1.2"}, "profile": {"pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.3", "ram_gib": 60.0}}) == 243.6)

# A first setup is listed while it fetches: bytes so far, total and speed, in
# the panel's status, until its manifest exists or remove dismisses it.
a.clear_installing("fresh")
a.begin_installing("fresh", {"name": "Fresh model", "vendor": "qwen"})
status = json.loads(Path(a.state_dir(), "status.json").read_text())
check("a setup shows as installing before it fetches anything", status["models"]["fresh"]["unit"] == "installing" and status["models"]["fresh"]["install"]["phase"] == "waiting" and status["models"]["fresh"]["name"] == "Fresh model")
progress = a.Progress(rid="fresh")
progress.samples = [(progress.started - 1, 0)]
progress.grow(4 * 2**20)
progress.published = 0
progress.add(3 * 2**20)
progress.published = 0
progress.pulled(2**20, 2 * 2**20)
row = json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["fresh"]
check("progress counts weights and image layers toward one bar", row["install"]["phase"] == "downloading" and row["install"]["done_bytes"] == 4 * 2**20 and row["install"]["total_bytes"] == 6 * 2**20, json.dumps(row["install"]))
check("progress reports a download speed", row["install"]["rate_bytes_s"] > 0)
with patch.object(a, "known_models", return_value=[]), patch.object(a, "unit_props", return_value={}):
  listed = a.status_record(a.load_catalog(), profile)
check("a full status rewrite keeps the setup's row", listed["models"]["fresh"]["unit"] == "installing")
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  a.write_status(a.load_catalog(), profile)
a.publish_installing("current", {"name": "Current", "phase": "downloading"})
check("setting up an installed model again leaves its row as it is", json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["current"]["unit"] != "installing")

a.clear_installing("timed")
a.begin_installing("timed", {"name": "Timed"}, 48.0)
check("a setup's row carries the catalog's load time for the panel's bar", json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["timed"]["expected_load_seconds"] == 48.0)
a.clear_installing("timed")
check("a first start runs against the catalog's load time", a.expected_load_seconds("never-started", {"measured_load_seconds": 48.0}) == 48.0)

# A failed first setup keeps its row with the reason, and remove dismisses it.
a.clear_installing(rid)
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "preflight"), patch.object(a, "build_artifacts", return_value=([], {"model": "model.gguf"})), patch.object(a, "materialize", side_effect=a.LocalError("Need 90 GiB free for downloads")), patch.object(a, "write_status"):
  raises("a failed setup reports its reason", lambda: a.cmd_use([rid]), "Need 90 GiB")
failed = a.load_installing(rid)
check("a failed setup stays listed with its reason", failed is not None and failed["phase"] == "failed" and failed["error"].startswith("Need 90 GiB"), json.dumps(failed))
check("a setup records its process for cancelling", failed["pid"] == os.getpid())
a.clear_installing(rid)
Path(a.model_manifest_path("dismissed")).unlink(missing_ok=True)
a.begin_installing("dismissed", {"name": "Dismissed"})
a.fail_installing("dismissed", "Download failed")
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  check("remove dismisses a failed setup", a.cmd_remove(["dismissed"]) == 0 and a.load_installing("dismissed") is None)
  check("remove stops the setup's unit first", ("stop", a.install_unit("dismissed")) in [call.args for call in stop.call_args_list])

# A setup waiting for another's lock is already cancellable, and stops if its
# record was removed while it waited.
class Waiting:
  def __init__(self, during=None):
    self.during = during
  def __enter__(self):
    self.seen = (a.load_installing(rid) or {}).get("pid")
    if self.during:
      self.during()
    return self
  def __exit__(self, *exc):
    return False
a.clear_installing(rid)
waiting = Waiting()
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=waiting), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  a.cmd_use([rid])
check("a setup records its process before it waits for the lock", waiting.seen == os.getpid())
a.update_installing(rid, pid=os.getppid())
second = Waiting()
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=second), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  a.cmd_use([rid])
check("a second use of the same recipe leaves the live owner's process in place while it waits", second.seen == os.getppid())
check("a queued use takes the setup over once it holds the lock", a.load_installing(rid)["pid"] == os.getpid())
a.update_installing(rid, pid=os.getppid())
class Interrupted(Waiting):
  def __enter__(self):
    raise KeyboardInterrupt
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=Interrupted()), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  try:
    a.cmd_use([rid])
  except KeyboardInterrupt:
    pass
check("interrupting a use still waiting leaves the active setup's record", (a.load_installing(rid) or {}).get("pid") == os.getppid())
a.update_installing(rid, image=runtime["image"])
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "Lock", return_value=Waiting()), patch.object(a, "load_model", return_value=None), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)), patch.object(a, "stop_setup_process"), patch.object(a, "write_status"), patch.object(a, "refresh_list_cache"), patch.object(a, "_remove_files", return_value=(0, [])) as removed:
  a.cmd_remove([rid])
check("cancelling a setup removes the image it pulled", removed.call_args.args == (rid, runtime["image"]))
with patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as rmi:
  a._remove_files(rid, runtime["image"])
check("removal drops a cancelled setup's image no installed model uses", rmi.call_args.args[0] == ["podman", "rmi", runtime["image"]])
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=Waiting(lambda: a.clear_installing(rid))), patch.object(a, "_use_setup", return_value=0) as setup, patch.object(a, "write_status"):
  raises("a setup removed while it waited stops", lambda: a.cmd_use([rid]), "cancelled")
  check("a cancelled setup fetches nothing", not setup.called and a.load_installing(rid) is None)

# Bytes an interrupted download left on disk count as done, but not as speed.
resumed = a.Progress()
resumed.grow(10 * 2**20)
resumed.resumed(4 * 2**20)
done_now, total_now, rate_now = resumed.snapshot()
check("a resumed download's saved bytes count toward the bar", done_now == 4 * 2**20 and total_now == 10 * 2**20)
check("a resumed download's saved bytes don't count as speed", rate_now == 0)

# A new attempt replaces a failed setup's record rather than inheriting it.
a.begin_installing("retried", {"name": "Retried"})
a.fail_installing("retried", "Download failed")
check("a failed setup can be set up again", a.installing_idle("retried"))
a.begin_installing("retried", {"name": "Retried"})
fresh_record = a.load_installing("retried")
check("a retry starts without the old failure", fresh_record["phase"] == "waiting" and fresh_record["error"] == "")
check("a setup under way isn't replaced", not a.installing_idle("retried"))
a.clear_installing("retried")

# Cancelling stops a model its setup already started and drops only its own partial downloads.
staging = Path(a.data_dir(), "staging")
staging.mkdir(parents=True, exist_ok=True)
mine, other = "a" * 64 + ".part", "b" * 64 + ".part"
(staging / mine).write_bytes(b"x" * 10)
(staging / other).write_bytes(b"y" * 10)
a.begin_installing("cancelled", {"name": "Cancelled"})
a.update_installing("cancelled", partials=[mine, "../escape.part"])
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  a.cmd_remove(["cancelled"])
check("cancelling a setup stops its model service", ("stop", a.unit_name("cancelled")) in [call.args for call in stop.call_args_list])
check("cancelling a setup drops only its own partial downloads", not (staging / mine).exists() and (staging / other).exists())

# --detach hands the setup to a transient user unit and returns.
a.clear_installing(rid)
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  check("a detached setup returns at once", a.detach_use(rid, SimpleNamespace(ctx=8192, gpu=None, seed=None, unqualified=False), cat.recipes[rid]) == 0)
  argv = run.call_args.args[0]
check("a detached setup runs use in its own user unit", argv[:3] == ["systemd-run", "--user", "--collect"] and f"--unit={a.install_unit(rid)}" in argv and argv[-4:-2] == ["use", rid] and argv[-2:] == ["--ctx", "8192"], " ".join(argv))
check("a detached setup is listed before its unit starts", a.load_installing(rid)["phase"] == "waiting")
a.clear_installing(rid)
with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_value", "OMARCHY_LOCAL_AI_CATALOG": "relative/catalog"}), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  a.detach_use(rid, SimpleNamespace(ctx=None, gpu=None, seed=None, unqualified=False), cat.recipes[rid])
  argv, env = run.call_args.args[0], run.call_args.kwargs["env"]
check("a detached setup never puts the token on a command line", not any("hf_secret_value" in arg for arg in argv) and "--setenv=HF_TOKEN" in argv and env["HF_TOKEN"] == "hf_secret_value")
check("a detached setup gets the catalog as an absolute path", env["OMARCHY_LOCAL_AI_CATALOG"] == os.path.abspath("relative/catalog"))
a.clear_installing(rid)

# Refresh shares one catalog/profile across both caches.
with patch.object(a, "load_catalog", return_value=cat) as load, patch.object(a, "machine_profile", return_value=profile) as machine, patch.object(a, "known_models", return_value=[]), patch.object(a, "running_models", return_value=set()):
  a.cmd_refresh(["--quiet"])
  check("refresh loads catalog and machine profile once", load.call_count == machine.call_count == 1)


# A machine over SSH runs what this machine can't: listed, placed and paired with here.
cat = a.load_catalog_dir(os.environ["ENGINE_ROOT"])
here = {"platform": "x86", "ram_class": 64, "ram_gib": 62.0, "gpus": []}
dgx = {"platform": "aarch64-gb10", "ram_class": 128, "ram_gib": 121.6, "gpus": [spark]}
bolt = "nemotron-3.5-bolt-tensorfold"
with patch.object(a, "load_machines", return_value={"me@dgx": dgx}):
  rows = {row["id"]: row for row in a.recipe_listing(cat, here)}
  check("a recipe only the machine runs is listed as available there", rows[bolt]["available"] and rows[bolt]["machine"] == "me@dgx" and "me@dgx" in rows[bolt]["summary"])
  check("what the machine can't run either stays unavailable", not rows["qwen3.8-27b-gufo"]["available"] and "machine" not in rows["qwen3.8-27b-gufo"]
      and rows["qwen3.8-27b-gufo"]["reason"].endswith("; on me@dgx: runtime gufo can't run on nvidia-gb10"))
  check("use places a recipe this machine can't run on the machine", a.machine_placement(cat, bolt, here) == "me@dgx" and a.machine_placement(cat, "best", here) == "me@dgx")
  check("use keeps what this machine runs here", a.machine_placement(cat, bolt, dgx) is None)
  raises("use --on names a machine that's set up", lambda: a.machine_placement(cat, bolt, here, on="other"), "isn't set up as a machine")

def mirrored(row):
  """Runs one round of mirror_installing against a machine whose status shows this row."""
  done = __import__("threading").Event()
  status = json.dumps({"models": {bolt: row}})
  with patch.object(a, "pair_ssh", side_effect=lambda *_, **__: (done.set(), SimpleNamespace(stdout=status))[1]), \
       patch.object(a, "update_installing") as installing:
    a.mirror_installing("me@dgx", bolt, SimpleNamespace(wait=lambda _: done.is_set()))
  return installing
installing = mirrored({"unit": "installing", "install": {"phase": "downloading", "done_bytes": 5, "total_bytes": 10, "rate_bytes_s": 2}})
check("a machine's download progress reaches this machine's setup row", installing.call_args.kwargs == {"phase": "downloading", "done_bytes": 5, "total_bytes": 10, "rate_bytes_s": 2})
check("a machine's model that's loading adds nothing to this machine's record", not mirrored({"unit": "activating", "expected_load_seconds": 50}).called)

def machine_use_run(share_code, args=None):
  """Runs machine_use for bolt on me@dgx, with every SSH step and pairing replaced; returns what was called."""
  calls = []
  def ssh(host, *argv, **kwargs):
    command = " ".join(argv)
    calls.append(command)
    return SimpleNamespace(returncode=share_code if command.startswith("share ") else 0, stdout="", stderr="no tailscale")
  with patch.object(a, "load_machines", return_value={"me@dgx": dgx}), patch.object(a, "load_model", return_value=None), patch.object(a, "installing_idle", return_value=True), \
       patch.object(a, "begin_installing"), patch.object(a, "update_installing"), patch.object(a, "clear_installing") as cleared, patch.object(a, "fail_installing") as failed, \
       patch.object(a, "owns_installing", return_value=True), patch.object(a, "sync_engine"), patch.object(a, "pair_ssh", side_effect=ssh), patch.object(a, "machine_engine", side_effect=ssh), \
       patch.object(a, "mirror_installing"), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0)), patch.object(a, "Lock"), patch.object(a, "cmd_pair") as pair:
    ns = SimpleNamespace(target=bolt, ctx=None, unqualified=False, detach=False, **(args or {}))
    try:
      a.machine_use("me@dgx", ns, cat)
      error = None
    except a.LocalError as caught:
      error = caught
  return calls, pair, cleared, failed, error
calls, pair, cleared, failed, error = machine_use_run(0)
check("a setup on a machine is shared there and paired with here, for that one model", error is None and f"share {bolt}" in calls and pair.call_args.args == (["me@dgx", bolt],) and cleared.called and not failed.called)
calls, pair, cleared, failed, error = machine_use_run(1)
check("a machine that can't share leaves the setup row failed, with the fix", error is not None and "Tailscale" in str(error) and f"pair me@dgx {bolt}" in str(error) and failed.called and not pair.called)

# ---- tailnet sharing

ts_status = {
  "BackendState": "Running",
  "Self": {"DNSName": "me.tail.ts.net.", "OS": "linux", "Online": True},
  "Peer": {
    "a": {"DNSName": "box.tail.ts.net.", "OS": "linux", "Online": True},
    "b": {"DNSName": "away.tail.ts.net.", "OS": "linux", "Online": False},
    "c": {"DNSName": "phone.tail.ts.net.", "OS": "android", "Online": True},
    "d": {"DNSName": "plain.tail.ts.net.", "OS": "linux", "Online": True},
    "e": {"DNSName": "flaky.tail.ts.net.", "OS": "linux", "Online": True},
  },
}
listing = {"schema": 1, "host": "box", "dns": "box.tail.ts.net", "ram_total_gib": 121.9, "models": [
  {"recipe": "big-model", "name": "Big\x1b[31m Model", "vendor": "qwen", "architecture": "moe", "ctx": 262144, "max_tokens": 10 ** 9, "input": ["text", "image", "audio"], "unit": "active",
   "observed": {"ram_used_gib": 80, "evil": "x"}, "since_ms": None, "decode_tok_s": 40},
  {"recipe": "../escape", "name": "bad"},
  {"recipe": "odd-state", "unit": "rebooting", "architecture": "<b>moe</b>"},
]}
clean = a.clean_listing(listing, "box.tail.ts.net")
check("a peer's listing keeps only recipe ids it can put in a path", [m["recipe"] for m in clean["models"]] == ["big-model", "odd-state"])
check("a peer's listing loses control characters and unknown inputs", clean["models"][0]["name"] == "Big[31m Model" and clean["models"][0]["input"] == ["text", "image"])
check("a peer's listing keeps only known states and measurements", clean["models"][1]["unit"] == "inactive" and "evil" not in clean["models"][0]["observed"])
check("a peer's listing keeps a known architecture and drops any other", clean["models"][0]["architecture"] == "moe" and clean["models"][1]["architecture"] == "")
check("a peer's reply cap is kept within its context, and a missing one is the default",
      clean["models"][0]["max_tokens"] == 262144 and clean["models"][1]["max_tokens"] == a.DEFAULT_MAX_OUTPUT_TOKENS)
raises("a listing of another schema is refused", lambda: a.clean_listing({"schema": 2, "models": []}, "box"), "Not an Omarchy")
check("a number too big for a float is no number", a._number(10 ** 400) is None)
class Huge:
  status = 200
  def __enter__(self): return self
  def __exit__(self, *args): return False
  def read(self, size=-1):
    return (b'{"schema": 1, "models": [' + b" " * (2 * a.HTTP_MAX_BYTES))[:size if size >= 0 else None]
with patch.object(a.urllib.request, "urlopen", return_value=Huge()):
  state, huge, _ = a.fetch_listing("big.tail.ts.net")
check("a peer's oversized listing is read only up to the cap and left unreachable", state == "unreachable" and huge is None)
with patch.object(a, "http_status", return_value=(200, b"[" * 100000 + b"]" * 100000)):
  state, deep, _ = a.fetch_listing("deep.tail.ts.net")
check("a peer's deeply nested listing leaves only that peer unreachable", state == "unreachable" and deep is None)
with patch.object(a.urllib.request, "urlopen", side_effect=__import__("http.client").client.BadStatusLine("garbage")):
  state, bad, _ = a.fetch_listing("bad.tail.ts.net")
check("a peer's malformed HTTP answer leaves only that peer unreachable", state == "unreachable" and bad is None)

answers = {
  "box.tail.ts.net": ("online", clean, ""),
  "plain.tail.ts.net": ("not_sharing", None, ""),
  "flaky.tail.ts.net": ("unreachable", None, "HTTP 502"),
}
previous = {"hosts": {
  "away": {"host": "away", "dns": "away.tail.ts.net", "last_seen_ms": 1_000_000, "ram_total_gib": 64},
  "flaky": {"host": "flaky", "dns": "flaky.tail.ts.net", "last_seen_ms": 900_000},
  "gone": {"host": "gone", "dns": "gone.tail.ts.net", "last_seen_ms": 2_000_000 - a.FORGET_UNPAIRED_MS},
}, "models": {
  "away/small-model": {"id": "away/small-model", "host": "away", "recipe": "small-model", "name": "Small", "unit": "active", "since_ms": 5},
  "flaky/mid-model": {"id": "flaky/mid-model", "host": "flaky", "recipe": "mid-model", "name": "Mid", "unit": "active"},
  "gone/old-model": {"id": "gone/old-model", "host": "gone", "recipe": "old-model", "unit": "active"},
}}
now = 2_000_000
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  net = a.survey_network(previous, now=now)
check("the survey lists a sharing peer's models by host/recipe", net["models"]["box/big-model"]["endpoint"] == "https://box.tail.ts.net/local-ai/big-model/v1" and net["models"]["box/big-model"]["offline"] == "")
check("a model its host has stopped is left off the list", "box/odd-state" not in net["models"])
check("the survey skips this machine, phones and peers sharing nothing", set(net["hosts"]) == {"box", "away", "flaky"})
check("sharing needs HTTPS certificates on the tailnet", net["can_share"] is False)
with patch.object(a, "tailscale_status", return_value={**ts_status, "CertDomains": ["me.tail.ts.net"]}), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  check("a tailnet with MagicDNS and HTTPS can share", a.survey_network(previous, now=now)["can_share"] is True)
check("a host that went offline keeps its models, marked offline", net["models"]["away/small-model"]["offline"] == "offline" and net["models"]["away/small-model"]["unit"] == "inactive" and net["hosts"]["away"]["state"] == "offline")
check("an online host that stopped answering is marked unreachable", net["models"]["flaky/mid-model"]["offline"] == "unreachable" and net["hosts"]["flaky"]["error"] == "HTTP 502")
check("offline hosts keep their last-seen time", net["hosts"]["away"]["last_seen_ms"] == 1_000_000 and net["hosts"]["box"]["last_seen_ms"] == now)
check("an unpaired host unseen for a week is dropped", "gone/old-model" not in net["models"])
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  later = a.survey_network(net, now=now + a.FORGET_UNPAIRED_MS)
check("an unpaired offline host is forgotten after a week", "away/small-model" not in later["models"] and "box/big-model" in later["models"])

# Pairing records survive; a paired host stays listed however long it's away.
a.ensure_dir(a.remotes_dir(), 0o700)
a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away"}))
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  later = a.survey_network(net, now=now + 10 * a.FORGET_UNPAIRED_MS)
check("a paired offline host keeps its models however long it's away", later["models"]["away/small-model"]["paired"] and later["models"]["away/small-model"]["offline"] == "offline")
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing") as fetch:
  alone = a.survey_network(net, now=now)
check("without tailscaled, paired hosts show offline and nothing is fetched", not fetch.called and alone["models"]["away/small-model"]["offline"] == "offline" and not alone["tailnet"])
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing") as fetch:
  blind = a.survey_network(previous, now=now)
check("without tailscaled, an unpaired host seen this week stays, offline", blind["models"].get("flaky/mid-model", {}).get("offline") == "offline" and "gone/old-model" not in blind["models"], repr(sorted(blind["models"])))
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away"])
forgotten = a.load_network()
check("forgetting a host drops its key and its models at once", not os.path.exists(a.remote_key_path("away")) and "away/small-model" not in forgotten["models"] and "box/big-model" in forgotten["models"])

# Pairing is per model: a host's key connects only the models it names.
def pair_away(**record):
  a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
  a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away", **record}))
def away_paired():
  with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
    return {mid: m["paired"] for mid, m in a.survey_network(later, now=now)["models"].items() if m["host"] == "away"}
pair_away()
check("a host paired whole connects every model it shares", away_paired() == {"away/small-model": True})
pair_away(models=["other-model"])
check("a host paired for one model leaves its others unpaired", away_paired() == {"away/small-model": False})
raises("a model that isn't connected can't be used", lambda: a.resolve_remote("away/small-model"), "isn't paired")
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a.subprocess, "run", side_effect=AssertionError("no SSH")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["away.tail.ts.net", "small-model"])
check("pairing another model of a paired host needs no SSH", a.load_remotes()["away"]["models"] == ["other-model", "small-model"] and away_paired() == {"away/small-model": True})
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting a model keeps the host's others and its key", a.load_remotes()["away"]["models"] == ["other-model"] and os.path.exists(a.remote_key_path("away")) and a.load_network()["models"]["away/small-model"]["paired"] is False)
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting a model that isn't connected leaves the host paired", a.load_remotes()["away"]["models"] == ["other-model"])
# A host paired whole: disconnecting one model needs no list of the others,
# so a model it shares later, or an empty survey, can't disconnect more.
pair_away()
a.write_atomic(a.network_path(), json.dumps({"hosts": {}, "models": {}}))
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing"):
  a.cmd_forget(["away/small-model"])
whole = a.load_remotes()["away"]
check("disconnecting a model of a host paired whole keeps the key and every other model, even with no survey",
      os.path.exists(a.remote_key_path("away")) and not a.connected(whole, "small-model") and a.connected(whole, "new-model"))
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a.subprocess, "run", side_effect=AssertionError("no SSH")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["away", "small-model"])
check("pairing it again reconnects it", a.connected(a.load_remotes()["away"], "small-model"))
pair_away(models=["small-model"])
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting the only model of a one-at-a-time pairing unpairs the host", "away" not in a.load_remotes() and not os.path.exists(a.remote_key_path("away")))

# The same host reached by another SSH name keeps what it connects.
pair_away(models=["other-model"])
export = json.dumps({"host": "away", "dns": "away.tail.ts.net", "key": "n" * 32})
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, export + "\n", "")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["alice@away", "small-model"])
again = a.load_remotes()["away"]
check("pairing a host again under another name adds to its models", again["models"] == ["other-model", "small-model"] and again["ssh"] == "alice@away")
# sshd answering but refusing every key isn't sshd being off: an agent left
# unstarted offers no key at all.
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "", "alice@new: Permission denied (publickey).\n")):
  raises("pairing with a host that refuses this machine's keys points at the SSH agent", lambda: a.cmd_pair(["alice@new"]), "SSH agent")
  # The panel shows the first line alone, so it names the machine short.
  raises("a pairing error's first line names the machine without user or domain", lambda: a.cmd_pair(["alice@new.tail1234.ts.net"]), "new refused this machine's SSH keys\n")
check("an address stays whole as a machine's name", a.short_host("dhh@100.90.162.80") == "100.90.162.80")
# The agents panel pairs with no terminal (stdin here is the test script), so
# ssh must ask nothing: a prompt would wait forever. A new host's key is taken.
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "", "Host key verification failed.\n")) as run:
  raises("pairing with a host whose key changed says how to forget the old one, by host alone", lambda: a.cmd_pair(["alice@new.tail1234.ts.net"]), "ssh-keygen -R new.tail1234.ts.net")
check("pairing without a terminal asks ssh for no prompts and takes a new host's key", {"BatchMode=yes", "StrictHostKeyChecking=accept-new"} <= set(run.call_args[0][0]), repr(run.call_args[0][0]))
a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away"}))
raises("a network model is named host/recipe", lambda: a.split_remote_id("box/../x"), "host/recipe")
raises("an SSH target can't smuggle options", lambda: a.ssh_command({"ssh": "-oProxyCommand=x", "dns": ""}, "share", "--export"), "Not an SSH target")
check("remote commands quote their arguments", a.ssh_command({"ssh": "box", "dns": "box.tail.ts.net"}, "share", "--export")[-2:] == ["box", "omarchy-agent-local share --export"])

# A paired, running remote model becomes an overlay with its own provider.
a.write_atomic(a.remote_key_path("box"), "b" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "box.json"), json.dumps({"host": "box", "dns": "box.tail.ts.net", "ssh": "box"}))
with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
  path, provider, model = a.pi_overlay("box/big-model")
  source = open(path).read()
  check("a remote pi overlay points at the host's tailnet endpoint and key", provider == "omarchy-local-box" and model == "big-model" and "https://box.tail.ts.net/local-ai/big-model/v1" in source and a.remote_key_path("box") in source and "b" * 32 not in source)
  oc = json.loads(open(a.opencode_overlay("box/big-model")[0]).read())["provider"]["omarchy-local-box"]
  check("a remote OpenCode overlay keeps the model's context and inputs", oc["models"]["big-model"]["limit"]["context"] == 262144 and oc["models"]["big-model"]["attachment"])
  raises("a remote model that isn't running can't be launched", lambda: a.pi_overlay("box/odd-state"), "isn't running")
raises("an unpaired host's model can't be launched", lambda: a.pi_overlay("plain/x"), "isn't paired")

# Paired hosts' models join pi's extension, one provider per host with its
# key, whether or not the host answers right now; unpaired ones don't.
seen = {"box/big-model": a.remote_record("box", "box.tail.ts.net", clean["models"][0], True, 128, 1),
        "box/odd-state": {**a.remote_record("box", "box.tail.ts.net", clean["models"][1], True, 128, 1), "offline": "offline"},
        "plain/x": a.remote_record("plain", "plain.tail.ts.net", {**clean["models"][0], "recipe": "x"}, False, 64, 1)}
a.write_atomic(a.network_path(), json.dumps({"hosts": {}, "models": seen}))
Path(a.pi_extension_path()).parent.parent.mkdir(parents=True, exist_ok=True)
with patch.object(a, "known_models", return_value=[]):
  a.sync_pi_extension()
  registered = open(a.pi_extension_path()).read()
  check("pi's extension lists a paired host's models under its own provider and key, offline ones too",
        registered.count("pi.registerProvider(") == 1 and '"omarchy-local-box"' in registered and '"big-model"' in registered and '"odd-state"' in registered
        and "https://box.tail.ts.net/local-ai/big-model/v1" in registered and a.remote_key_path("box") in registered and "b" * 32 not in registered)
  check("pi's extension leaves out an unpaired host's models", "plain" not in registered)
  with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
    source = open(a.pi_overlay("box/big-model")[0]).read()
  check("a remote pi overlay keeps the rest of its host's models, the launched one first",
        source.index('"big-model"') < source.index('"odd-state"') and "omarchy-local\"," not in source)
  # Qwen Code is launched with the endpoint and model, and reads the key itself.
  with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
    path, provider, model = a.qwen_overlay("box/big-model")
  qwen = json.loads(open(path).read())
  check("a Qwen Code overlay carries the endpoint, model, limits and key file, never the key",
        (provider, model) == ("omarchy-local-box", "big-model") and qwen["baseUrl"] == "https://box.tail.ts.net/local-ai/big-model/v1"
        and qwen["model"] == "big-model" and qwen["maxTokens"] == 262144
        and qwen["keyFile"] == a.remote_key_path("box") and qwen["contextWindow"] == 262144 and "b" * 32 not in open(path).read())
  with patch.object(a, "survey_network", return_value={"hosts": {}, "models": {}}):
    a.write_network()
  check("a network write syncs pi's extension, dropping models it no longer sees", not os.path.exists(a.pi_extension_path()))
check("the overlay command offers Qwen Code", "qwen" in a.OVERLAYS)

# Sharing: routes follow shares.json, the listing never carries the key.
handlers = {}
def fake_tailscale(*args, timeout=10):
  if args[:2] == ("serve", "status"):
    return SimpleNamespace(returncode=0, stdout=json.dumps({"Web": {"me.tail.ts.net:443": {"Handlers": dict(handlers)}}}), stderr="")
  path, target = args[-2], args[-1]
  if target == "off":
    handlers.pop(path, None)
  else:
    handlers[path] = {"Text": target[5:]} if target.startswith("text:") else {"Proxy": target}
  return SimpleNamespace(returncode=0, stdout="", stderr="")
share_model = dict(manifest, recipe_id="current")
a.save_model(share_model, started=False)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={"ActiveState": "active"}), patch.object(a, "free_memory", return_value=(64, 48)), \
     patch.object(a, "tailscale", side_effect=fake_tailscale), patch.object(a, "tailscale_status", return_value=ts_status), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")) as proxy_systemctl:
  record = a.write_status(cat, profile)
  check("nothing touches Tailscale Serve until a model is shared", handlers == {} and not record["models"]["current"]["shared"])
  a.save_shares(["current"])
  record = a.write_status(cat, profile)
  served = json.loads(handlers["/local-ai/models.json"]["Text"])
  check("a shared model is served through the share proxy", handlers["/local-ai/current"] == {"Proxy": f"http://127.0.0.1:{a.SHARE_PROXY_PORT}/current"} and record["models"]["current"]["shared"])
  check("sharing starts the share proxy", ("start", a.SHARE_PROXY_UNIT) in [call.args for call in proxy_systemctl.call_args_list])
  check("the listing names the model and its state, never the key", served["models"][0]["recipe"] == "current" and served["models"][0]["unit"] == "active" and "key" not in served and "key" not in served["models"][0])
  handlers["/other"] = {"Proxy": "http://127.0.0.1:9"}
  a.save_shares([])
  a.write_status(cat, profile)
  check("unsharing removes only our routes", handlers == {"/other": {"Proxy": "http://127.0.0.1:9"}} and not os.path.exists(a.published_path()))
  check("unsharing the last model stops the share proxy", proxy_systemctl.call_args_list[-1].args == ("stop", a.SHARE_PROXY_UNIT))
def refuse(*args, timeout=10):
  if args[:2] == ("serve", "status"):
    return SimpleNamespace(returncode=0, stdout="{}", stderr="")
  return SimpleNamespace(returncode=1, stdout="", stderr="sending serve config: 401 Unauthorized: must be root, or be an operator")
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={"ActiveState": "active"}), patch.object(a, "free_memory", return_value=(64, 48)), \
     patch.object(a, "tailscale", side_effect=refuse), patch.object(a, "tailscale_status", return_value=ts_status), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")):
  a.save_shares(["current"])
  record = a.write_status(cat, profile)
  check("a refused share is reported with the fix", record["share_error"] == a.OPERATOR_HINT)
a.save_shares([])
def hangs(output, error=b""):
  def run(*args, **kwargs):
    raise subprocess.TimeoutExpired(args[0], 10, output=output, stderr=error)
  return run
with patch.object(a.shutil, "which", return_value="/usr/bin/tailscale"):
  with patch.object(a.subprocess, "run", side_effect=hangs(b"", b"Serve is not enabled on your tailnet.\nTo enable, visit:\n\n  https://login.tailscale.com/f/serve?node=n1\n")):
    try:
      a.tailscale("serve", "--bg", "x")
      said = ""
    except a.LocalError as error:
      said = str(error)
  check("a tailscale serve that waits for Serve to be enabled says so, with the link", "Serve is not enabled on your tailnet. To enable, visit: https://login.tailscale.com/f/serve?node=n1" in said, said)
  with patch.object(a.subprocess, "run", side_effect=hangs(None)):
    try:
      a.tailscale("serve", "--bg", "x")
      said = ""
    except a.LocalError as error:
      said = str(error)
  check("one that printed nothing still just didn't answer", said == "tailscale serve didn't answer", said)

# The share proxy: a request the model accepts counts its machine for the
# window, a refused one doesn't, and a streamed answer arrives as it's sent.
import http.client
import http.server
import threading

release = threading.Event()

class FakeModel(http.server.BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"
  def log_message(self, *args):
    pass
  def do_POST(self):
    if self.headers.get("Transfer-Encoding") == "chunked":
      body = b""
      while size := int(self.rfile.readline().strip(), 16):
        body += self.rfile.read(size)
        self.rfile.readline()
      self.rfile.readline()
    else:
      body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
    if self.headers.get("Authorization") != "Bearer k":
      self.send_response(401)
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Transfer-Encoding", "chunked")
    self.end_headers()
    for index, part in enumerate((b"data: " + body + b"\n\n", b"data: [DONE]\n\n")):
      # The second part waits until the first has reached the client, so a
      # proxy that held the answer back until it ended would never finish.
      if index:
        release.wait(10)
      self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
      self.wfile.flush()
    self.wfile.write(b"0\r\n\r\n")

def serve(handler):
  server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
  server.daemon_threads = True
  threading.Thread(target=server.serve_forever, daemon=True).start()
  return server

now = [1_000_000]
tracker = a.ClientTracker(a.clients_path(), window_ms=1000, clock=lambda: now[0])
model = serve(FakeModel)
proxy = serve(a.make_share_proxy(tracker, ports=lambda: {"shared": model.server_address[1], "private": model.server_address[1]}, shares=lambda: ["shared"]))

def ask(path, key="k", client="100.64.0.2"):
  connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
  connection.request("POST", path, body=b"hello", headers={"Authorization": f"Bearer {key}", "X-Forwarded-For": client, "Content-Type": "application/json"})
  answer = connection.getresponse()
  body = answer.read()
  connection.close()
  return answer.status, body

def counts():
  return json.loads(open(a.clients_path()).read())["models"]

connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
connection.request("POST", "/shared/v1/chat/completions", body=iter([b"hel", b"lo"]), encode_chunked=True, headers={"Authorization": "Bearer k", "X-Forwarded-For": "100.64.0.2"})
answer = connection.getresponse()
first = answer.read1(65536)
release.set()
rest = answer.read()
connection.close()
check("the proxy streams each part of the answer as the model sends it", answer.status == 200 and first == b"data: hello\n\n" and rest == b"data: [DONE]\n\n", f"{answer.status} {first!r} {rest!r}")
release.clear()
release.set()
status, body = ask("/shared/v1/chat/completions")
check("a request with a length passes its body through", status == 200 and body == b"data: hello\n\ndata: [DONE]\n\n", f"{status} {body!r}")
check("an accepted request counts its machine", counts() == {"shared": 1})
ask("/shared/v1/chat/completions")
check("the same machine counts once", counts() == {"shared": 1})
status, _ = ask("/shared/v1/chat/completions", key="wrong", client="100.64.0.3")
check("a request the model refuses doesn't count", status == 401 and counts() == {"shared": 1})
ask("/shared/v1/chat/completions", client="100.64.0.3")
check("a second machine counts as a second user", counts() == {"shared": 2})
status, _ = ask("/private/v1/chat/completions", client="100.64.0.4")
check("a model that isn't shared isn't reachable through the proxy", status == 404 and counts() == {"shared": 2})
now[0] += 600
ask("/shared/v1/chat/completions", client="100.64.0.3")
now[0] += 500
check("a machine idle past the window stops counting", tracker.expire() is not None and counts() == {"shared": 1})
now[0] += 600
check("once everyone is idle nothing counts and nothing is pending", tracker.expire() is None and counts() == {})
proxy.shutdown()
model.shutdown()


# ---------------------------------------------------------------- user recipes
import contextlib
import io

curated_id = "qwen3.8-27b-gufo"
curated_text = Path(os.environ["ENGINE_ROOT"], "recipes", curated_id + ".toml").read_text()
mine = Path(os.environ["HOME"]) / "mine"
mine.mkdir()
os.environ["OMARCHY_LOCAL_AI_CATALOG"] = os.environ["ENGINE_ROOT"]


def write_mine(rid, text):
  path = mine / f"{rid}.toml"
  path.write_text(text)
  return str(path)


def run(command, *argv):
  out, err = io.StringIO(), io.StringIO()
  with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    try:
      code = command(list(argv))
    except a.LocalError as error:
      code = 1
      err.write(str(error))
  return code, out.getvalue(), err.getvalue()


def variants_as(text, source):
  return re.sub(r"(\[\[variant\]\]\n)", rf'\1source = "{source}"\n', text)


good = write_mine("my-model", variants_as(curated_text, "estimated"))
check("the user recipe folder lives in the state dir beside models and machines", a.user_recipes_dir() == os.path.join(a.state_dir(), "recipes"))
code, out, err = run(a.cmd_recipe, "validate", good)
check("recipe validate accepts a good file and installs nothing", code == 0 and not os.path.exists(a.user_recipes_dir()), out + err)
code, _, _ = run(a.cmd_recipe, "add", good)
check("recipe add installs the file under its id", code == 0 and os.path.isfile(os.path.join(a.user_recipes_dir(), "my-model.toml")))
cat = a.load_catalog()
check("the catalog loads curated recipes, then user ones", "my-model" in cat.recipes and curated_id in cat.recipes and cat.user == {"my-model"} and cat.curated.isdisjoint(cat.user))
code, out, _ = run(a.cmd_recipe, "list", "--json")
with patch.object(a, "machine_profile", return_value=profile), patch.object(a, "running_models", return_value=set()), patch.object(a, "load_machines", return_value={}):
  code, out, err = run(a.cmd_recipe, "list", "--json")
rows = {row["id"]: row for row in json.loads(out)}
check("recipe list --json marks user recipes true and curated false", code == 0 and rows["my-model"]["user"] is True and rows[curated_id]["user"] is False, out + err)
check("a user recipe survives an engine update: it is not in the engine's folder", not Path(os.environ["ENGINE_ROOT"], "recipes", "my-model.toml").exists() and a.load_catalog().user == {"my-model"})

# a clash with a curated id never shadows
clash = write_mine(curated_id, curated_text)
for command in ("validate", "add"):
  code, out, err = run(a.cmd_recipe, command, clash)
  check(f"recipe {command} rejects an id that clashes with a curated recipe, naming it", code == 1 and f"clashes with the curated recipe '{curated_id}'" in out + err, out + err)
shadow = Path(a.user_recipes_dir(), f"{curated_id}.toml")
shadow.write_text(curated_text.replace("name =", 'name = "Shadow" #', 1))
shadowed = a.load_catalog()
check("a user file with a curated id is skipped and never shadows", shadowed.recipes[curated_id].get("name") != "Shadow" and curated_id in shadowed.user_errors and curated_id not in shadowed.user)
shadow.unlink()

# an invalid recipe gets the errors validate_recipe gives a curated one
broken = variants_as(curated_text, "estimated").replace('revision = "', 'revision = "z', 1)
bad = write_mine("bad-one", broken)
expected = a.validate_recipe("bad-one", tomllib.loads(broken), a.load_catalog().runtimes, os.environ["ENGINE_ROOT"], user=True)
code, out, _ = run(a.cmd_recipe, "validate", bad)
check("an invalid recipe is rejected with every error validate_recipe gives", code == 1 and expected and out.splitlines() == expected, out)
code, _, _ = run(a.cmd_recipe, "add", bad)
check("recipe add of an invalid recipe installs nothing", code == 1 and not os.path.exists(os.path.join(a.user_recipes_dir(), "bad-one.toml")))
looser = tomllib.loads(curated_text)
looser["server"] = {"args": ["--host", "0.0.0.0"]}
check("a user recipe has no looser tier: reserved args are still refused", any("reserved" in e for e in a.validate_recipe("x", looser, a.load_catalog().runtimes, os.environ["ENGINE_ROOT"], user=True)))
code, out, _ = run(a.cmd_recipe, "validate", write_mine("not-toml", "this is = = broken"))
check("recipe validate reports a file that isn't TOML", code == 1 and "not valid TOML" in out, out)
code, out, _ = run(a.cmd_recipe, "validate", str(mine / "missing.toml"))
check("recipe validate reports a missing file", code == 1 and "can't be read" in out, out)
Path(a.user_recipes_dir(), "hand-broken.toml").write_text(broken)
tolerant = a.load_catalog()
check("a hand-placed invalid user recipe is skipped without breaking the catalog", "hand-broken" in tolerant.user_errors and "hand-broken" not in tolerant.recipes and "my-model" in tolerant.recipes)
Path(a.user_recipes_dir(), "hand-broken.toml").unlink()

# a measured file is the catalog's evidence: a user recipe may omit it, a curated one may not
no_measured = re.sub(r', measured = "[^"]*"|\nmeasured = "[^"]*"', "", curated_text)
check("the fixture drops the measured file", "measured" not in no_measured)
check("a user recipe may omit a hardware entry's measured file", not a.validate_recipe("x", tomllib.loads(no_measured), a.load_catalog().runtimes, os.environ["ENGINE_ROOT"], user=True))
check("a curated recipe still needs it", any("measured" in e for e in a.validate_recipe("x", tomllib.loads(no_measured), a.load_catalog().runtimes, os.environ["ENGINE_ROOT"])))

# remove
code, out, err = run(a.cmd_recipe, "remove", curated_id)
check("recipe remove refuses a curated recipe", code == 1 and "curated" in err, err)
code, _, err = run(a.cmd_recipe, "remove", "nothing-here")
check("recipe remove refuses an unknown id", code == 1 and "No recipe of yours" in err, err)
Path(a.models_dir()).mkdir(parents=True, exist_ok=True)
Path(a.model_manifest_path("my-model")).write_text(json.dumps({"recipe": {"name": "My model"}}))
code, _, err = run(a.cmd_recipe, "remove", "my-model")
check("recipe remove of an in-use recipe fails naming the model", code == 1 and "my-model (My model)" in err and os.path.isfile(os.path.join(a.user_recipes_dir(), "my-model.toml")), err)
code, _, err = run(a.cmd_recipe, "add", good)
check("recipe add over an in-use recipe fails naming the model too", code == 1 and "my-model (My model)" in err, err)
os.unlink(a.model_manifest_path("my-model"))
code, _, err = run(a.cmd_recipe, "remove", "my-model")
check("recipe remove of an unused recipe succeeds", code == 0 and not os.path.exists(os.path.join(a.user_recipes_dir(), "my-model.toml")) and "my-model" not in a.load_catalog().recipes, err)

# variant sources
for source, ok in (("measured", True), ("estimated", True), ("drafted", True), ("guessed", False)):
  errors = a.validate_recipe("x", tomllib.loads(variants_as(curated_text, source)), a.load_catalog().runtimes, os.environ["ENGINE_ROOT"], user=True)
  check(f"a variant source of '{source}' is {'accepted' if ok else 'rejected'}", (not errors) == ok, "\n".join(errors))
check("a variant without a source is measured", a.fits(variant, profile, gpu, "model")[0] and a.fits(dict(variant, source="measured"), profile, gpu, "model")[0])
drafted_fit = a.fits(dict(variant, source="drafted"), profile, gpu, "model")
check("a drafted variant never counts as fitting", not drafted_fit[0] and "draft" in drafted_fit[1])
estimated = dict(variant, source="estimated")
check("an estimated variant doesn't fit before it is confirmed", not a.fits(estimated, profile, gpu, "model")[0])
est_add = write_mine("est-model", variants_as(curated_text, "estimated"))
run(a.cmd_recipe, "add", est_add)
code, out, err = run(a.cmd_recipe, "confirm", "est-model")
est_cat = a.load_catalog()
est_variants = est_cat.recipes["est-model"]["variant"]
check("recipe confirm records the estimated variants once", code == 0 and all(a.variant_key(v) in a.load_confirmed()["est-model"] for v in est_variants), out + err)
small = dict(est_variants[0], peak_ram_gib=1, peak_gpu_gib=1)
Path(a.confirmed_path()).write_text(json.dumps({"est-model": [a.variant_key(small)]}))
check("a confirmed estimated variant fits, one with other numbers doesn't", a.fits(small, profile, gpu, "est-model")[0] and not a.fits(dict(small, peak_ram_gib=2), profile, gpu, "est-model")[0])
picked, reason = a.pick_variant({"variant": [dict(small, source="drafted")]}, profile, gpu, None, "est-model")
check("pick_variant skips a drafted variant and says why", picked is None and "draft" in reason, reason)
# every path a user recipe reaches counts a confirmed estimated variant as fitting
big = dict(profile, ram_class=128, ram_gib=128)
est_recipe = est_cat.recipes["est-model"]
Path(a.confirmed_path()).write_text(json.dumps({"est-model": sorted({a.variant_key(v) for v in est_recipe["variant"]})}))
est_gpu = a.recipe_gpu(big, est_recipe)
check("a confirmed estimated variant fits through fits", a.fits(est_recipe["variant"][0], big, est_gpu, "est-model")[0])
check("a confirmed estimated variant fits through pick_variant", a.pick_variant(est_recipe, big, est_gpu, None, "est-model")[0] is not None)
check("a confirmed estimated variant fits through compatibility", a.compatibility(est_cat, "est-model", big, est_gpu)[0] is not None)
est_cat.best = {"class": [{"gpu": est_gpu["family"], "ram_class": 128, "recipes": ["est-model"]}]}
check("a confirmed estimated variant fits through resolve_best", a.resolve_best(est_cat, big, est_gpu)[0] == "est-model")
with patch.object(a, "machine_profile", return_value=big), patch.object(a, "running_models", return_value=set()), patch.object(a, "load_machines", return_value={}):
  code, out, err = run(a.cmd_recipe, "list", "--json")
check("a confirmed estimated variant is available through recipe list", {r["id"]: r for r in json.loads(out)}["est-model"]["available"] is True, out + err)
check("a confirmed estimated variant runs here through runs_on", a.runs_on(est_cat, "est-model", big))
Path(a.confirmed_path()).write_text("{}")
check("an unconfirmed estimated variant fits through none of them", a.compatibility(est_cat, "est-model", big, est_gpu)[0] is None and not a.runs_on(est_cat, "est-model", big))
try:
  a.fits(estimated, profile, gpu)
except TypeError:
  check("fits can't be called without a rid", True)
else:
  check("fits can't be called without a rid", False)

# a confirmation is keyed by a stable form of the numbers
toml_a = tomllib.loads("ctx = 4096\npeak_ram_gib = 0.1\npeak_gpu_gib = 16\n")
toml_b = tomllib.loads("ctx = 4096\npeak_ram_gib = 0.10\npeak_gpu_gib = 16.0\n")
check("0.1 and 0.10 give one variant key", a.variant_key(toml_a) == a.variant_key(toml_b))
check("a real change, however small, changes the key", a.variant_key(dict(toml_a, peak_ram_gib=12.3456781)) != a.variant_key(dict(toml_a, peak_ram_gib=12.3456782)))

# recipe add: one read, the same bytes validated and written, no symlinks, no claimed measurements
for label, text in (("measured", variants_as(curated_text, "measured")), ("no source", curated_text)):
  claim = write_mine("claims", text)
  code, out, err = run(a.cmd_recipe, "add", claim)
  check(f"recipe add rejects a variant with {label}, pointing at estimated or drafted", code == 1 and 'source = "estimated"' in err and '"drafted"' in err and not os.path.exists(os.path.join(a.user_recipes_dir(), "claims.toml")), err)
  code, out, err = run(a.cmd_recipe, "validate", claim)
  check(f"recipe validate says the same for {label}", code == 1 and 'source = "estimated"' in out, out)
code, _, _ = run(a.cmd_recipe, "add", write_mine("drafty", variants_as(curated_text, "drafted")))
check("recipe add accepts drafted", code == 0)
run(a.cmd_recipe, "remove", "drafty")
swap_good = variants_as(curated_text, "estimated")
swap_path = write_mine("swapped", swap_good)
opened = []
real_open = os.open
def counting_open(path, *args, **kwargs):
  if str(path) == swap_path:
    opened.append(path)
    fd = real_open(path, *args, **kwargs)
    Path(swap_path + ".new").write_text(curated_text)
    os.replace(swap_path + ".new", swap_path)  # swapped right after the open
    return fd
  return real_open(path, *args, **kwargs)
with patch.object(os, "open", counting_open):
  code, out, err = run(a.cmd_recipe, "add", swap_path)
installed = Path(a.user_recipes_dir(), "swapped.toml")
check("recipe add opens the file once", len(opened) == 1, str(opened))
check("recipe add installs exactly the bytes it validated", code == 0 and installed.read_text() == swap_good, err)
run(a.cmd_recipe, "remove", "swapped")
link = mine / "linked.toml"
link.symlink_to(swap_path)
code, _, err = run(a.cmd_recipe, "add", str(link))
check("recipe add refuses a symlink source", code == 1 and "symlink" in err and not os.path.exists(os.path.join(a.user_recipes_dir(), "linked.toml")), err)

# update survival: a fresh engine copy in a new place sees the same user recipes and confirmations
fresh_engine = Path(os.environ["HOME"]) / "engine-after-update"
subprocess.run(["cp", "-r", os.environ["ENGINE_ROOT"], str(fresh_engine)], check=True)
subprocess.run(["rm", "-rf", str(fresh_engine / ".git")], check=True)
Path(a.confirmed_path()).write_text(json.dumps({"est-model": ["k"]}))
with patch.dict(os.environ, {"OMARCHY_LOCAL_AI_CATALOG": str(fresh_engine)}):
  after = a.load_catalog()
  check("user recipes and confirmations survive an engine update", "est-model" in after.user and a.load_confirmed() == {"est-model": ["k"]})
check("neither lives under the engine's folder", not a.state_dir().startswith(os.environ["ENGINE_ROOT"]) and a.confirmed_path().startswith(a.state_dir()) and a.user_recipes_dir().startswith(a.state_dir()))
Path(a.confirmed_path()).write_text("{}")

code, _, _ = run(a.cmd_recipe, "confirm", curated_id)
check("recipe confirm only takes a user recipe's estimated variants", code == 1)
run(a.cmd_recipe, "remove", "est-model")
check("recipe remove drops the recipe's confirmation", "est-model" not in a.load_confirmed())
check("recipe with no subcommand prints usage and fails", run(a.cmd_recipe)[0] == 1)

# ---------------------------------------------------------------- measure
# No GPU, network, ssh or machine: the runtime, the bench and memory are fakes.
import threading

measure_id = "measure-me"
drafted_text = variants_as(curated_text, "drafted")
measure_path = Path(a.user_recipes_dir(), measure_id + ".toml")
write_mine(measure_id, drafted_text)
run(a.cmd_recipe, "add", str(mine / f"{measure_id}.toml"))
measure_ctx = 131072
fake = {"running": False, "ram": 100.0, "gpu": 100.0, "fail_load": False, "calls": [], "bench_fail": False}
GB = 2**30


def fake_free_memory(_gpu):
  return fake["ram"], fake["gpu"]


def fake_use_setup(rid, variant, args, catalog, profile_, gpu_, fresh, measured_load=None, sampler=None):
  fake["calls"].append(("setup", rid, variant["ctx"]))
  if fake["fail_load"]:
    raise a.LocalError("The model didn't start: out of memory\nlog tail")
  sampler.start()
  fake["ram"], fake["gpu"] = 61.5, 70.25
  fake["running"] = True
  return 0


def fake_bench_request(prompt, model, key, max_tokens=512):
  fake["calls"].append(("bench", len(prompt)))
  if fake["bench_fail"]:
    raise OSError("connection reset")
  fake["ram"], fake["gpu"] = 52.25, 58.5   # the peak: KV filled during the long prompt
  long_one = len(prompt) > 50000
  return {"prompt_tokens": 20000 if long_one else 40, "completion_tokens": 512, "ttft_s": 10.0, "prefill_tok_s": 2000.0 if long_one else 50.0, "decode_tok_s": 21.5 if long_one else 25.0, "total_s": 30.0}


def reset_fake(**over):
  fake.update({"running": False, "ram": 100.0, "gpu": 100.0, "fail_load": False, "calls": [], "bench_fail": False}, **over)


a_gpu = {**gpu, "memory_gib": 100.0, "sysfs": "/nonexistent"}
a_profile = {**profile, "ram_gib": 128, "ram_class": 128, "gpus": [a_gpu]}
fake_files = [{"repo": "o/r", "revision": "a" * 40, "path": "m.gguf", "dest": "m.gguf", "size": 3 * GB, "license": "mit", "sha256": "b" * 64},
              {"repo": "o/r", "revision": "a" * 40, "path": "d.gguf", "dest": "d.gguf", "size": GB // 2, "license": "mit", "sha256": "c" * 64}]


def measure(*argv):
  with patch.object(a, "machine_profile", return_value=a_profile), patch.object(a, "recipe_gpu", return_value=a_gpu), \
       patch.object(a, "build_artifacts", return_value=(fake_files, {})), patch.object(a, "prepare_host"), \
       patch.object(a, "free_memory", fake_free_memory), patch.object(a, "_use_setup", fake_use_setup), \
       patch.object(a, "bench_request", fake_bench_request), patch.object(a, "ensure_key", return_value="k"), \
       patch.object(a, "unit_active", lambda rid: fake["running"]), patch.object(a, "systemctl"), \
       patch.object(a, "write_status"), patch.object(a, "refresh_list_cache"), patch.object(a, "load_model", return_value=None):
    return run(a.cmd_measure, *argv)


before = measure_path.read_bytes()
code, out, err = measure(measure_id, "--ctx", str(measure_ctx))
check("measure without --yes states the download size and what it loads", code == 0 and "3.5 GiB" in out and str(measure_ctx) in out and "--yes" in out, out + err)
check("measure without --yes changes nothing and runs nothing", measure_path.read_bytes() == before and fake["calls"] == [])
help_out = io.StringIO()
with contextlib.redirect_stdout(help_out), contextlib.suppress(SystemExit):
  a.cmd_measure(["--help"])
check("measure documents --ctx and --yes", "--ctx" in help_out.getvalue() and "--yes" in help_out.getvalue(), help_out.getvalue())
code, out, err = measure(curated_id, "--ctx", "131072", "--yes")
check("measure refuses a curated recipe with a clear message", code == 1 and "curated" in err and fake["calls"] == [], err)
code, out, err = measure("no-such-recipe", "--ctx", "131072", "--yes")
check("measure refuses an unknown recipe", code == 1 and "No recipe" in err, err)
code, out, err = measure(measure_id, "--ctx", "999", "--yes")
check("measure needs a variant for that ctx and says which exist", code == 1 and "131072" in err and measure_path.read_bytes() == before and fake["calls"] == [], err)

reset_fake(fail_load=True)
code, out, err = measure(measure_id, "--ctx", str(measure_ctx), "--yes")
check("a failed load reports the engine's own error", code == 1 and "The model didn't start: out of memory" in err, err)
check("a failed load leaves the recipe file byte-identical", measure_path.read_bytes() == before)

reset_fake(bench_fail=True)
code, out, err = measure(measure_id, "--ctx", str(measure_ctx), "--yes")
check("a failed bench leaves the recipe file byte-identical", code == 1 and measure_path.read_bytes() == before, err)

reset_fake()
code, out, err = measure(measure_id, "--ctx", str(measure_ctx), "--yes")
check("measure --yes succeeds", code == 0 and "measured" in out, out + err)
done = {v["ctx"]: v for v in tomllib.loads(measure_path.read_text())["variant"]}
v = done[measure_ctx]
check("the variant is now source = measured", v["source"] == "measured", str(v))
check("peak RAM is the baseline minus the lowest free RAM seen (100 - 52.25)", v["peak_ram_gib"] == 47.8, str(v))
check("peak GPU is the baseline minus the lowest free GPU seen (100 - 58.5)", v["peak_gpu_gib"] == 41.5, str(v))
check("decode and prefill speed come from the long prompt", v["decode_tok_s"] == 21.5 and v["prefill_tok_s"] == 2000.0, str(v))
check("the other variant is untouched", done[262144]["source"] == "drafted" and done[262144]["peak_ram_gib"] == tomllib.loads(drafted_text)["variant"][0]["peak_ram_gib"])
check("the measured recipe still validates and fits as measured", not a.user_recipe_errors(measure_id, tomllib.loads(measure_path.read_text()), a.load_catalog_dir(os.environ["ENGINE_ROOT"]).recipes, a.load_catalog_dir(os.environ["ENGINE_ROOT"]).runtimes, os.environ["ENGINE_ROOT"])
      and a.fits(v, a_profile, a_gpu, measure_id)[0])
check("a recipe rewrite keeps every other field", {k: x for k, x in tomllib.loads(measure_path.read_text()).items() if k != "variant"} == {k: x for k, x in tomllib.loads(drafted_text).items() if k != "variant"})

reset_fake()
estimated_text = variants_as(curated_text, "estimated")
write_mine(measure_id, estimated_text)
run(a.cmd_recipe, "add", str(mine / f"{measure_id}.toml"))
code, out, err = measure(measure_id, "--ctx", "262144", "--yes")
check("a re-measure replaces an estimated variant", code == 0 and {v["ctx"]: v for v in tomllib.loads(measure_path.read_text())["variant"]}[262144]["source"] == "measured", out + err)
check("a measured variant needs no confirmation", a.fits({v["ctx"]: v for v in tomllib.loads(measure_path.read_text())["variant"]}[262144], a_profile, a_gpu, measure_id)[0])
reset_fake()
code, out, err = measure(measure_id, "--ctx", "262144", "--yes")
check("a measured variant can be measured again", code == 0, out + err)

# the long prompt never exceeds the context
small_prompt = a.bench_context("x" * 105000, 8192)
check("bench_context shortens the corpus for a small context", len(small_prompt) < 105000 * 8192 / 31705 and len(small_prompt) > 0)
check("bench_context uses (nearly) the whole corpus when the context is big", len(a.bench_context("x" * 105000, 262144)) > 100000)

# use on a user recipe needs --yes
reset_fake()
mine_text = variants_as(curated_text, "estimated")
write_mine("use-me", mine_text)
run(a.cmd_recipe, "add", str(mine / "use-me.toml"))
run(a.cmd_recipe, "confirm", "use-me")
used = []
with patch.object(a, "machine_profile", return_value=a_profile), patch.object(a, "recipe_gpu", return_value=a_gpu), patch.object(a, "prepare_host", side_effect=lambda g: used.append("prepare")):
  code, out, err = run(a.cmd_use, "use-me")
check("use refuses a user recipe without --yes, naming the flag", code == 1 and "--yes" in err and used == [], err)
with patch.object(a, "machine_profile", return_value=a_profile), patch.object(a, "recipe_gpu", return_value=a_gpu), patch.object(a, "prepare_host", side_effect=lambda g: used.append("prepare")), \
     patch.object(a, "_use_setup", return_value=0), patch.object(a, "load_measurement", return_value={}), patch.object(a, "load_model", return_value={"recipe_id": "use-me"}):
  code, out, err = run(a.cmd_use, "use-me", "--yes", "--unqualified")
check("use --yes lets a user recipe through", code == 0 and used == ["prepare"], out + err)

# measure holds the lock, and says so
dry_code, dry_out, dry_err = measure(measure_id, "--ctx", str(measure_ctx))
check("measure says up front that other commands wait for it", dry_code == 0 and "other omarchy agent local command waits" in dry_out and "30 minutes" in dry_out, dry_out + dry_err)
messages, entered = [], threading.Event()


def blocked_command():
  with patch.object(a, "say", messages.append), a.Lock():
    entered.set()


with a.Lock(note=f"a measuring run of {measure_id}"):
  waiter = threading.Thread(target=blocked_command, daemon=True)
  with patch.object(a, "say", messages.append):
    waiter.start()
    for _ in range(200):
      if messages:
        break
      time.sleep(0.01)
waiter.join(5)
check("a blocked command's wait message says a measuring run holds the lock", any("measuring run" in m and measure_id in m for m in messages), str(messages))
check("a blocked command goes on once the lock is free", entered.is_set())
messages.clear()
with patch.object(a, "say", messages.append), a.Lock():
  pass
check("nothing is said when the lock is free", messages == [], str(messages))
with a.Lock():
  pass
with a.Lock(note="stale"):
  pass
check("the note is cleared when the lock is released", Path(a.state_dir(), "lock").read_text() == "")

# toml_dumps: values TOML can't hold are a LocalError; DEL is escaped
import datetime
raises("toml_dumps names a datetime it can't write as a LocalError", lambda: a.toml_dumps(tomllib.loads("when = 2026-01-02T03:04:05")), "datetime")
raises("toml_dumps names a date it can't write as a LocalError", lambda: a.toml_dumps({"table": {"day": datetime.date(2026, 1, 2)}}), "date")
raises("toml_dumps refuses a set as a LocalError", lambda: a.toml_dumps({"x": {1}}), "set")
check("toml_dumps escapes DEL in values and keys", tomllib.loads(a.toml_dumps({"k\x7fey": "a\x7fb"})) == {"k\x7fey": "a\x7fb"} and "\x7f" not in a.toml_dumps({"k\x7fey": "a\x7fb"}))

# tools/bench samples with MemorySampler
bench_globals = bench.cmd_bench.__globals__
bench_sampler = []
real_sampler = a.MemorySampler


class RecordingSampler(real_sampler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bench_sampler.append(self)


readings = iter([(50.0, 50.0), (40.0, 40.0)] + [(45.0, 45.0)] * 1000)
active_model = {"recipe_id": "model", "gpu": gpu["family"], "files": [], "fileset": "f", "image": "i", "variant": {"ctx": 4096},
                "profile": {"platform": "x86", "ram_class": 64}}
bench_out = io.StringIO()
with patch.dict(bench_globals, {"MemorySampler": RecordingSampler, "resolve_model": lambda *_, **__: active_model, "choose_gpu": lambda *_: gpu,
                                "machine_profile": lambda: profile, "run_bench": lambda *args, **kwargs: ({}, {}, {}), "key_path": lambda: str(mine / "key"),
                                "free_memory": lambda _gpu: next(readings)}), \
     patch.object(a, "free_memory", lambda _gpu: next(readings)), contextlib.redirect_stdout(bench_out):
  (mine / "key").write_text("k")
  bench.cmd_bench([])
check("tools/bench samples memory with the engine's MemorySampler", len(bench_sampler) == 1 and "min_available_during_bench_gib = 40.0" in bench_out.getvalue(), bench_out.getvalue())

PY
