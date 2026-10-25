# LLM quantization tools

This is a comparison of tools that are practical for quantizing open-weight language models. “MoE support” means the project can load or transform mixture-of-experts (MoE) checkpoints; it does not mean every MoE architecture is automatically supported. Always run a small generation and routing smoke test with the target model.

| Tool | Primary formats / methods | MoE support | License | Notes |
|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF; K-quants, IQ-quants, Q8_0 | Yes, for architectures its HF-to-GGUF converter recognizes, including OLMoE in current releases | MIT | Best fit for local GGUF inference. `llama-quantize --override-tensor` can keep router and output tensors at a higher precision. |
| [LLM Compressor](https://github.com/vllm-project/llm-compressor) | GPTQ, AWQ, SmoothQuant, FP8, sparse quantization; `compressed-tensors` | Architecture-dependent Transformers support; validate OLMoE before a production run | Apache-2.0 | Recipe-based quantization and a good option for vLLM deployments. Can ignore named layers/modules in a recipe. |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ (primarily 4-bit) | Architecture-dependent; model adapters/configuration may be required | MIT | Fast weight-only 4-bit workflow, mainly aimed at GPU inference. |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | GPTQ, 2–8-bit weight-only | Architecture-dependent; confirm OLMoE support in the installed release | Apache-2.0 | Maintained GPTQ-oriented toolkit with backend integrations. |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | LLM.int8(), NF4/FP4 | Yes for supported Transformers modules; expert/router placement should be explicitly checked | MIT | Common for loading/fine-tuning models at 8-bit or 4-bit rather than producing GGUF files. |
| [torchao](https://github.com/pytorch/ao) | PyTorch weight-only and dynamic quantization | Generic PyTorch-module support; OLMoE requires model-level validation | BSD-3-Clause | PyTorch-native research/production toolkit; useful when serving through PyTorch. |
| [Intel Neural Compressor](https://github.com/intel/neural-compressor) | PTQ, GPTQ, AWQ, SmoothQuant | Architecture-dependent; test the OLMoE model path | Apache-2.0 | Broad CPU/XPU-focused optimization toolkit. |
| [Optimum Quanto](https://github.com/huggingface/optimum-quanto) | INT8/INT4/FP8 PyTorch quantization | Generic module support; router/expert coverage needs verification | Apache-2.0 | Lightweight Hugging Face/PyTorch integration. |
| [HQQ](https://github.com/mobiusml/hqq) | Half-quadratic quantization, low-bit weights | Architecture-dependent; custom MoE modules need validation | Apache-2.0 | Research-oriented low-bit method, with backends for efficient inference. |
| [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | FP8, INT8, INT4/AWQ, pruning | Yes for supported model integrations; verify the relevant OLMoE release | Apache-2.0 | Pairs with TensorRT-LLM on NVIDIA GPUs; NVIDIA runtime components have their own terms. |

## Selected workflow: llama.cpp + OLMoE

`quantize_olmoe.sh` downloads `allenai/OLMoE-1B-7B-0924` by default, builds the current `llama.cpp`, converts a BF16 GGUF baseline, and creates:

- `Q8_0`
- `Q6_K`
- `Q5_K_M`
- `Q4_K_M`
- `Q3_K_M`

Every variant retains `output.weight` and OLMoE expert-router tensors matching `blk.<n>.ffn_gate_inp.weight` as **BF16**. The script verifies those tensors in each generated GGUF and exits with an error if the expected tensors are absent or not BF16.

Run everything with one command:

```bash
./quantize_olmoe.sh
```

Optional environment variables:

```bash
MODEL_ID=allenai/OLMoE-1B-7B-0924 OUTPUT_DIR="$PWD/artifacts" ./quantize_olmoe.sh
```

Requirements: `git`, `cmake`, a C/C++ compiler, Python 3.10+, and enough disk space for the original checkpoint, BF16 GGUF, and five quantized GGUF files. Set `HF_TOKEN` when downloading a gated Hugging Face model. The script intentionally fails with a clear message if the installed `llama-quantize` does not support selective `--override-tensor` quantization; silently quantizing router/output tensors would violate the precision requirement.
