# G9's eval_mini on OLMoE GGUF files

Runs G9's **eval_mini** profile (`teams/09-evaluation/harness/run_eval_mini.py` in upstream main) on bf16 and the
llama.cpp formats, with **per-item logs**, and compares each format with bf16 **item by item**.

| File | What |
|---|---|
| [`llamacpp_lm.py`](llamacpp_lm.py) | lm-evaluation-harness model class: a GGUF file run in-process with llama-cpp-python |
| [`run_mini_gguf.py`](run_mini_gguf.py) | one GGUF through G9's profile; writes G9's record format + per-item `*.samples.jsonl` |
| [`paired.py`](paired.py) | each format vs bf16 on the same items: difference, 95% CI, McNemar's exact test |
| [`eval_mini.ipynb`](eval_mini.ipynb) | Kaggle or Colab notebook: build, convert, run every format, compare; resumable; carries its own copy of the three files above |
| [`build_notebook.py`](build_notebook.py) | rebuilds the notebook from the three `.py` files (edit those, then rebuild) |
| [`combine.py`](combine.py) | merges the zips from several members, checks they used identical code and settings, writes scores + paired comparison |
| [`extract_results.py`](extract_results.py) | recovers the results zip that the notebook's last cell prints (for a Colab kernel attached from VS Code) |

## Team run: split the formats, combine the results

bf16 is done (Kaggle 2×T4, 28 Oct; ARC-C 46.8, HellaSwag 66.0, MMLU 53.7, GSM8K 10.5, TruthfulQA-mc2 37.3). The six
quantized formats are split between members, two each (~2 h per member on Kaggle, compile included):

| Formats | Run by (GitHub handle) | Status |
|---|---|---|
| `Q8_0`, `Q6_K` | | |
| `Q5_K_M`, `Q4_K_M` | | |
| `Q3_K_M`, `Q2_K` | s9k96 | done 29 Oct |

**For each member:** import `eval_mini.ipynb` into Kaggle (Accelerator **GPU T4 x2**, Internet **on**), set
`FORMATS` and `RUN_BY` in the settings cell and nothing else, then **Save Version → Save & Run All (Commit)**.
Every format is checked by `test_llamacpp_lm.py` before it is scored; a failure stops the run.

**Committing your results:** download `eval_mini_out/runs_<formats>.zip` from the version's Output tab, unzip it to
`results/raw/runs_<formats>/` (the folder holds your format folders, `environment.json` and `logs/`), and open a PR
that adds only that folder (branch from `G12-Week1`, or from `shubham_eval_mini` until it is merged). Do not edit
other runs' folders. Each file stays well under pr-guard's 5 MB limit (a run of two formats is about 4 MB).

| Run folder | Formats | Run by | Machine | Date |
|---|---|---|---|---|
| `results/raw/runs_bf16/` | bf16 | s9k96 | Kaggle 2×T4 | 28 Oct |
| `results/raw/runs_Q3_K_M-Q2_K/` | Q3_K_M, Q2_K | s9k96 | Kaggle 2×T4 | 29 Oct |

**Combining:** `python combine.py results/raw/* --out results/combined` (zips work too). It accepts a run only if
the model revision, profile, seed, n_ctx, lm_eval and llama-cpp-python versions, G9's runner and the full item counts
match the bf16 run, and all runs carry the same notebook fingerprint (`NOTEBOOK_VERSION`); test runs (`LIMIT` set)
are refused. Hardware may differ and is recorded per format in `provenance.json`.

Do not edit the notebook between members' runs: any change to the embedded code changes the fingerprint.

## Why not llama-server (or lm_eval's own GGUF backend)

lm_eval scores multiple choice (ARC, HellaSwag, MMLU, TruthfulQA) by the log-probability of each answer, so it
needs the model's log-probabilities for the **prompt** tokens. Checked 28 Oct:

- **llama-server (b11669)** ignores `echo=True` and returns log-probabilities only for generated tokens. lm_eval's
  `local-completions` and `gguf` backends therefore cannot score multiple choice (GSM8K alone would work).
- **llama-cpp-python's server (0.3.36)** returns prompt log-probabilities **shifted by one position**:
  log p(" Paris" | "The capital of France is") comes back as −8.58 (the distribution *after* " Paris") instead of
  −0.49. lm_eval's `gguf` backend sums these without any error and produces plausible but meaningless scores.
- **transformers' GGUF loader (5.20.2)** does not support the OLMoE architecture (it does support Qwen3-MoE).

`llamacpp_lm.py` keeps every position's logits and reads token i's log-probability from position i−1:
log p(" Paris" | …) = −0.49, the same as greedy generation reports.

## Checks done on an Apple M5 (llama-cpp-python 0.3.36, lm_eval 0.4.13, G9's pins)

- 20 items each, ARC-Challenge / HellaSwag, `acc_norm`: Q8_0 0.45 / 0.65, Q2_K 0.25 / 0.55. Q8_0 is well above
  chance (0.25), Q2_K lower; at 20 items the gaps are within noise.
- MMLU (2 per subject = 114 items), TruthfulQA-mc2 and GSM8K (generation) run end to end; every task writes one
  per-item row per item (GSM8K one per answer filter). Re-running gives identical scores.
- The prefix cache (a multiple-choice question is evaluated once, not once per answer) changes log-probabilities
  by at most 0.004 compared with evaluating every answer from scratch; greedy flags are identical.
- `paired.py`'s McNemar p-values match hand-computed cases.
- The whole notebook, run locally in its test mode (`EVAL_MINI_LOCAL_TEST=1`: two formats, 2 items per task):
  every cell runs, all outputs are written (per-format `summary.json` + `*.samples.jsonl`, `scores.csv`,
  `paired.md`, `environment.json`, logs, a zip), and a second run skips the finished formats.

- The converter's separate environment builds from llama.cpp's requirements file (32 s) with the versions that
  made the team's bf16 file.

Not yet checked: the GPU-only steps (CUDA build of llama-cpp-python, bf16 on the T4s) and timings.

## Run

```bash
# one format, on any machine with llama-cpp-python 0.3.36 + lm_eval 0.4.13
python run_mini_gguf.py --gguf OLMoE-1B-7B-0924-Q8_0.gguf --format Q8_0 --out runs \
    --g9-harness <upstream main>/teams/09-evaluation/harness/run_eval_mini.py
python paired.py runs --base bf16
```

**Kaggle (recommended):** upload `eval_mini.ipynb` (Create → New Notebook → File → Import Notebook), set
Accelerator **GPU T4 x2** and Internet **on**. It starts with `LIMIT = 5`, a quick end-to-end test: run all cells
interactively (results in `/kaggle/working/eval_mini_out/runs_limit5/`). For the full run set `LIMIT = None` and use
**Save Version → Save & Run All (Commit)**: it runs in the background and the results zip appears in that version's
Output tab. Big files go to `/kaggle/tmp`, outside the 20 GB output limit; bf16 splits across the two T4s.

**Colab:** T4 runtime, Run all. Output goes to Drive if it mounts; otherwise (e.g. a Colab kernel attached from VS
Code, where files stay on Google's VM) the last cell prints the results zip as base64; save the notebook, then
`python extract_results.py eval_mini.ipynb` rebuilds it locally (it checks the length, so a truncated output is
reported, not silently decoded).

Every step stops the run if it fails, and finished tasks are skipped on the next run in the same session.

Comparing with bf16 needs bf16 itself to run through this same backend: switching backends moves scores by
about as much as the quantization effects being measured (G9's noise note: ~2 points per 100 items from a kernel
change alone).
