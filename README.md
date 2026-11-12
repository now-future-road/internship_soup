# Qwen2.5-1.5B-Instruct LIMA SFT Finetuning

This repository contains the configuration, data-cleaning workflow, and validation routines for Supervised Fine-Tuning (SFT) of **Qwen2.5-1.5B-Instruct** using the **LIMA** dataset on a Tesla T4 GPU (using `soup-cli`).

## 🚀 Project Overview
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset:** LIMA (filtered & structured to Alpaca format)
- **Task Type:** SFT (Supervised Fine-Tuning)
- **Optimization:** LoRA (r=16, alpha=32) + NF4 Quantization

## 📦 File Structures
- `soup.yaml`: High-fidelity training configuration parameters.
- `requirements.txt`: Package dependencies.
- `notebook.ipynb`: Core step-by-step training pipeline, validation suite, and quantization analysis.

## 🔍 Validation Suite Outcomes
- **Weight Divergence:** Passed — Mean absolute weight shift of `0.00041289` across adapted target layers (`self_attn`, `mlp`) on the final `./output/checkpoint-600` merge (`4.75e-6` when averaged across all 1.5B parameters including frozen `embed_tokens`/`lm_head`). Saved adapter artifact: `./output/checkpoint-600/adapter_model.safetensors` (merged locally to `./output/merged_model`).
- **Prompt Masking:** Passed (Prompt tokens correctly masked with `-100`, zero loss leakage).
- **Template Compliance:** Passed (Validated with Qwen ChatML markers `<|im_start|>` & `<|im_end|>`).
