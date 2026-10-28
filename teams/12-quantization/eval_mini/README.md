# G9's eval_mini on OLMoE GGUF files

Runs G9's **eval_mini** profile (`teams/09-evaluation/harness/run_eval_mini.py` in upstream main) on bf16 and the
llama.cpp formats, with **per-item logs**, and compares each format with bf16 **item by item**.

| File | What |
|---|---|
| [`llamacpp_lm.py`](llamacpp_lm.py) | lm-evaluation-harness model class: a GGUF file run in-process with llama-cpp-python |
| [`run_mini_gguf.py`](run_mini_gguf.py) | one GGUF through G9's profile; writes G9's record format + per-item `*.samples.jsonl` |
| [`paired.py`](paired.py) | each format vs bf16 on the same items: difference, 95% CI, McNemar's exact test |
| [`eval_mini_colab.ipynb`](eval_mini_colab.ipynb) | Colab T4 notebook: build, convert, run every format, compare; resumable, results to Drive |

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

Not yet checked: the Colab T4 run itself (GPU build, bf16 fit, timings).

## Run

```bash
# one format, on any machine with llama-cpp-python 0.3.36 + lm_eval 0.4.13
python run_mini_gguf.py --gguf OLMoE-1B-7B-0924-Q8_0.gguf --format Q8_0 --out runs \
    --g9-harness <upstream main>/teams/09-evaluation/harness/run_eval_mini.py
python paired.py runs --base bf16
```

On Colab: open `eval_mini_colab.ipynb`, choose a T4 runtime, Run all. Set `LIMIT = 5` first for a quick end-to-end
test. Comparing with bf16 needs bf16 itself to run through this same backend: switching backends moves scores by
about as much as the quantization effects being measured (G9's noise note: ~2 points per 100 items from a kernel
change alone).
