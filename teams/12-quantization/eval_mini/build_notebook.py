#!/usr/bin/env python3
"""Build eval_mini_colab.ipynb from the .py files next to it.

    python build_notebook.py

The notebook carries its own copy of llamacpp_lm.py, run_mini_gguf.py and paired.py (written to disk by its
first code cell), so it runs without this branch being pushed. The .py files stay the source of truth: edit
them, then rebuild. G9's run_eval_mini.py is still fetched from upstream main, where it is published.

Local test (same cells, no Colab): EVAL_MINI_LOCAL_TEST=1 jupyter nbconvert --execute eval_mini_colab.ipynb
uses the GGUFs in ~/.cache/era-v5-quant, two formats and a few items per task.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ["llamacpp_lm.py", "run_mini_gguf.py", "paired.py"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
md("""
# G12: G9's eval_mini on OLMoE GGUF files (Colab T4)

Runs G9's **mini** profile (ARC-C 500, HellaSwag 400, MMLU 570, GSM8K 200, TruthfulQA-mc2 200; G9's shots, limits
and seed) on **bf16 and six llama.cpp formats**, all through one backend (llama-cpp-python in-process; see
`llamacpp_lm.py` for why not llama-server), with **per-item logs**, then compares each format with bf16 item by
item (`paired.py`).

**Runtime:** T4 GPU. Then **Run all**. Every step stops the run if it fails.

- Output goes to Google Drive when it can be mounted, else to `/content/eval_mini_out` (download the zip at the
  end). Each task is saved as soon as it finishes; **Run all again after a disconnect** and finished tasks are
  skipped. The compiled llama-cpp-python is cached next to the output, so a second session skips the compile.
- First run: keep `LIMIT = 5` (~30-40 min, mostly the one-time compile and model conversion). Then set
  `LIMIT = None` for the full run (estimate: ~25 min per quantized format, ~1-1.5 h for bf16).
"""),
code("""
# --- settings -----------------------------------------------------------------------------------------------
LIMIT     = 5          # items per task for a test run (MMLU: per subject); None = G9's full mini profile
FORMATS   = ["bf16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]   # bf16 first: it is the baseline
BASE      = "bf16"
MODEL, REVISION = "allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5"  # team pin
LLAMA_TAG = "b11669"           # llama.cpp build of the team's conversion/quantization
LLAMA_CPP_PYTHON = "0.3.36"    # in-process log-probabilities checked correct with this version
G9_RAW = "https://raw.githubusercontent.com/theschoolofai/era-v5-lab/main/teams/09-evaluation/harness/run_eval_mini.py"
DRIVE_DIR = "/content/drive/MyDrive/era-v5-g12/eval_mini"   # used when Drive mounts
LOCAL_OUT = "/content/eval_mini_out"                        # used otherwise
WORK      = "/content/work"                                 # model files, rebuilt per session
BF16_FALLBACK_GPU_LAYERS = 12  # of 16 layers, only if bf16 does not fit on the GPU

import os
if os.environ.get("EVAL_MINI_LOCAL_TEST"):   # dry run of these same cells on a Mac (see build_notebook.py)
    LIMIT, FORMATS, BASE = 2, ["Q8_0", "Q2_K"], "Q8_0"
    LOCAL_OUT = os.environ.get("EVAL_MINI_OUT", "/tmp/eval_mini_local")
    WORK = os.path.expanduser("~/.cache/era-v5-quant")
"""),
code("""
import datetime, glob, json, os, platform, shutil, subprocess, sys, time
LOCAL_TEST = bool(os.environ.get("EVAL_MINI_LOCAL_TEST"))
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

OUT = LOCAL_OUT
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT = DRIVE_DIR
    except Exception as e:  # e.g. not available from an editor-attached runtime
        print(f"Drive not mounted ({e!r}); output goes to {LOCAL_OUT}. DOWNLOAD THE ZIP AT THE END.")
for d in (OUT, f"{OUT}/logs", WORK):
    os.makedirs(d, exist_ok=True)
CODE = f"{WORK}/eval_mini_code"; os.makedirs(CODE, exist_ok=True)

def sh(cmd, log=None):
    \"\"\"Run a shell command; stream its output (and append it to a log); stop the notebook if it fails.\"\"\"
    print("$", cmd, flush=True)
    full = f"set -o pipefail; ( {cmd} ) 2>&1" + (f" | tee -a {log}" if log else "")
    rc = subprocess.run(full, shell=True, executable="/bin/bash").returncode
    if rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}" + (f"; see {log}" if log else ""))

ENV = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "in_colab": IN_COLAB, "local_test": LOCAL_TEST,
       "output_dir": OUT, "settings": {k: globals()[k] for k in ("LIMIT", "FORMATS", "BASE", "MODEL", "REVISION",
       "LLAMA_TAG", "LLAMA_CPP_PYTHON", "BF16_FALLBACK_GPU_LAYERS")}, "python": platform.python_version(), "formats": {}}
def save_env():
    json.dump(ENV, open(f"{OUT}/environment.json", "w"), indent=1)
if shutil.which("nvidia-smi"):
    ENV["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                                shell=True, capture_output=True, text=True).stdout.strip()
    sh("nvidia-smi")
sh(f"df -h {WORK} | tail -1")
save_env()
print("output:", OUT)
"""),
"__SOURCE_CELL__",
code("""
# llama-cpp-python with CUDA, lm_eval and datasets at G9's pins. The CUDA build is compiled once (~15-25 min on
# Colab's 2 CPUs) into a wheel kept in OUT/wheels; later sessions install it in seconds.
if not LOCAL_TEST:
    wheels = glob.glob(f"{OUT}/wheels/llama_cpp_python-{LLAMA_CPP_PYTHON}-*.whl")
    if not wheels:
        os.makedirs(f"{OUT}/wheels", exist_ok=True)
        sh(f'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip wheel --no-deps --no-cache-dir '
           f'-w {OUT}/wheels llama-cpp-python=={LLAMA_CPP_PYTHON}', log=f"{OUT}/logs/build_llama_cpp_python.log")
        wheels = glob.glob(f"{OUT}/wheels/llama_cpp_python-{LLAMA_CPP_PYTHON}-*.whl")
    sh(f"pip install -q {wheels[0]} lm_eval==0.4.13 datasets==5.1.0", log=f"{OUT}/logs/pip.log")
import importlib, llama_cpp, lm_eval, datasets
gpu_ok = llama_cpp.llama_supports_gpu_offload()
if IN_COLAB and not gpu_ok:
    raise RuntimeError("llama-cpp-python has no GPU offload: delete OUT/wheels and rerun to rebuild it with CUDA")
ENV["packages"] = {"llama_cpp_python": llama_cpp.__version__, "lm_eval": lm_eval.__version__,
                   "datasets": datasets.__version__, "gpu_offload": gpu_ok}
save_env(); print(ENV["packages"])
"""),
code("""
# G9's runner (profile, task list, record format) from upstream main; its sha256 goes into every summary
sh(f"curl -fsSL {G9_RAW} -o {CODE}/run_eval_mini.py")
sys.path.insert(0, CODE)
"""),
code("""
# bf16 GGUF: the llama.cpp converter at the team's build, pinned checkpoint; llama-quantize built CPU-only (~5 min)
NAME = MODEL.split("/")[-1]
BF16 = f"{WORK}/{NAME}-bf16.gguf"
QUANTIZE = shutil.which("llama-quantize") if LOCAL_TEST else f"{WORK}/llama.cpp/build/bin/llama-quantize"
if not LOCAL_TEST and not os.path.exists(QUANTIZE):
    if not os.path.exists(f"{WORK}/llama.cpp"):
        sh(f"git clone -q --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    sh(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_CUDA=OFF "
       f"&& cmake --build {WORK}/llama.cpp/build -j --target llama-quantize", log=f"{OUT}/logs/build_llama_quantize.log")
if not os.path.exists(BF16):
    sh(f"pip install -q {WORK}/llama.cpp/gguf-py sentencepiece", log=f"{OUT}/logs/pip.log")
    from huggingface_hub import snapshot_download
    hf = snapshot_download(MODEL, revision=REVISION, local_dir=f"{WORK}/hf",
                           allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    sh(f"python {WORK}/llama.cpp/convert_hf_to_gguf.py {hf} --outtype bf16 --outfile {BF16}", log=f"{OUT}/logs/convert.log")
    shutil.rmtree(f"{WORK}/hf", ignore_errors=True)
ENV["bf16_gguf_bytes"] = os.path.getsize(BF16); save_env()
print(BF16, round(os.path.getsize(BF16) / 1e9, 2), "GB")
"""),
code("""
# every format: quantize (same rule as scripts/quantize_eval.py: output layer Q8_0 below 8 bits), run G9's profile,
# save after each task. A format whose five tasks are all in its summary.json is skipped.
TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k", "truthfulqa_mc2"]
RUNS = f"{OUT}/runs" + (f"_limit{LIMIT}" if LIMIT else "")     # test runs never mix with the full run

def finished(fmt):
    p = f"{RUNS}/{fmt}/summary.json"
    return os.path.exists(p) and {r["task"] for r in json.load(open(p))["records"]} >= set(TASKS)

def run_format(fmt, gguf, gpu_layers):
    sh(f"cd {CODE} && {sys.executable} run_mini_gguf.py --gguf {gguf} --format {fmt} --out {RUNS} "
       f"--g9-harness {CODE}/run_eval_mini.py --n-gpu-layers {gpu_layers}" + (f" --limit {LIMIT}" if LIMIT else ""),
       log=f"{OUT}/logs/run_{fmt}.log")

for fmt in FORMATS:
    if finished(fmt):
        print(f"{fmt}: all tasks done, skipped"); continue
    t0 = time.time()
    gguf = BF16 if fmt == "bf16" else f"{WORK}/{NAME}-{fmt}.gguf"
    made_here = False
    if not os.path.exists(gguf):
        extra = "" if fmt == "Q8_0" else "--output-tensor-type q8_0"
        sh(f"{QUANTIZE} {extra} {BF16} {gguf} {fmt}", log=f"{OUT}/logs/quantize_{fmt}.log"); made_here = True
    gpu_layers = -1
    try:
        run_format(fmt, gguf, gpu_layers)
    except RuntimeError:
        if fmt != "bf16":
            raise
        gpu_layers = BF16_FALLBACK_GPU_LAYERS   # bf16 did not fit with every layer on the GPU
        print(f"bf16 failed with all layers on the GPU; retrying with {gpu_layers} GPU layers")
        run_format(fmt, gguf, gpu_layers)
    ENV["formats"][fmt] = {"gpu_layers": gpu_layers, "gguf_bytes": os.path.getsize(gguf),
                           "wall_clock_s": round(time.time() - t0), "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    save_env()
    if made_here and fmt != "bf16" and not LOCAL_TEST:
        os.remove(gguf)   # keep the local disk small
"""),
code("""
# results: G9-primary scores per format, the paired comparison with the base, a zip of everything
import pandas as pd
rows = []
for p in sorted(glob.glob(f"{RUNS}/*/summary.json")):
    s = json.load(open(p))
    for r in s["records"]:
        if r["primary"]:
            rows.append({"format": s["run_meta"]["format"], "task": r["task"], "points": round(100 * r["value"], 1),
                         "stderr": round(100 * (r["stderr"] or 0), 1), "n": r["n_items"], "seconds": r["wall_clock_s"]})
long = pd.DataFrame(rows)
long.to_csv(f"{RUNS}/scores_long.csv", index=False)
scores = long.pivot_table(index="format", columns="task", values="points").reindex([f for f in FORMATS if f in set(long.format)])
scores.to_csv(f"{RUNS}/scores.csv")
print(scores.to_string())
sh(f"{sys.executable} {CODE}/paired.py {RUNS} --base {BASE} --out {RUNS}/paired.md")
ENV["finished"] = datetime.datetime.now().isoformat(timespec="seconds"); save_env()
shutil.copy(f"{OUT}/environment.json", f"{RUNS}/environment.json")
shutil.copytree(f"{OUT}/logs", f"{RUNS}/logs", dirs_exist_ok=True)
zip_path = shutil.make_archive(RUNS, "zip", RUNS)
print("\\nEverything is in", RUNS, "and", zip_path)
n_files = sum(len(f) for _, _, f in os.walk(RUNS)); print(n_files, "files:", sorted(os.listdir(RUNS)))
"""),
code("""
# bring the results back to the machine running VS Code: the zip, as base64 text in this cell's output. With a
# Colab kernel every file above lives on Google's VM; cell outputs are what VS Code keeps when you save the
# notebook. Locally: python extract_results.py eval_mini_colab.ipynb  -> writes the zip next to the notebook.
import base64
blob = base64.b64encode(open(zip_path, "rb").read()).decode()
print(f"EVAL_MINI_ZIP_BEGIN {os.path.basename(zip_path)} {len(blob)}")
for i in range(0, len(blob), 4000):
    print(blob[i:i + 4000])
print("EVAL_MINI_ZIP_END")
"""),
md("""
## Send back

**Save the notebook in VS Code** (the last cell's output carries the results), then locally run
`python extract_results.py eval_mini_colab.ipynb`. It writes the zip (`runs_limit5.zip` for the test run, `runs.zip`
for the full run): per format a `summary.json`
(G9's record format) and one `*.samples.jsonl` per task; `scores.csv`, `scores_long.csv`, `paired.md`,
`environment.json` (GPU, versions, settings, per-format GPU layers and time) and `logs/`. From Drive, or if Drive
did not mount, download it from the runtime before the session ends.
"""),
]


def build() -> Path:
    src = {name: (HERE / name).read_text() for name in SOURCES}
    writer = ["# the evaluation code (llamacpp_lm.py, run_mini_gguf.py, paired.py), embedded by build_notebook.py",
              "SOURCES = " + json.dumps(src, ensure_ascii=True, indent=0),
              "for _name, _text in SOURCES.items():",
              "    open(f\"{CODE}/{_name}\", \"w\").write(_text)",
              "print(\"wrote\", sorted(SOURCES), \"to\", CODE)"]
    cells = [code("\n".join(writer)) if c == "__SOURCE_CELL__" else c for c in CELLS]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"}}}
    out = HERE / "eval_mini_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n")
    return out


if __name__ == "__main__":
    print("wrote", build())
