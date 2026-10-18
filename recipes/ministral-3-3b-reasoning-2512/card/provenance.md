No model was re-exported from PyTorch. The graphs are Mistral's: the Reasoning and Instruct checkpoints share one architecture and one config (`text_config` and `vision_config` are identical), so every `.onnx` file here is a copy of the same file in the Instruct ONNX repo, with one fix in the decoders (below), and the weights in the `.onnx_data` files are new. Every initializer keeps the reference's file, offset, length, dtype and shape, so the external-data layout is the same down to the byte offset.

- Base checkpoint: [mistralai/Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) at `4a36357c811bf511a7b625d132e12f22408aac91` (`model-0000{1,2}-of-00002.safetensors`, bf16).
- Graphs and layout: [mistralai/Ministral-3-3B-Instruct-2512-ONNX](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX) at `e1128d389145f96bb1c3c2625cb8ca255ffa7034`. Their node names, ops (`GroupQueryAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization`, `MatMulNBits`, `GatherBlockQuantized`) and the YaRN rope caches come from an onnxruntime-genai-style model builder; `cos_cache` and `sin_cache` depend on the config only and are copied.
- From the Reasoning checkpoint: `tokenizer.json`, `special_tokens_map.json`, `chat_template.jinja` (also inlined into `tokenizer_config.json`, as in the reference), `processor_config.json`, `generation_config.json` (with Mistral's recommended `temperature` 0.7 and `top_p` 0.95 added from the base model card), `SYSTEM_PROMPT.txt`. From the reference: `config.json` (it carries `transformers.js_config` and the text `model_type` transformers.js knows) and `preprocessor_config.json` (the checkpoint has none).

Weights, per initializer type, with the method the reference was made with:

| Reference initializers | Written as |
|---|---|
| plain fp32 / fp16 | bf16 → fp32 (exact) → fp16 for the fp16 graphs. Linear weights transposed to (in, out); `lm_head` is the tied `embed_tokens` table. |
| `*_Q4` + `*_scales` (decoder `q4`, `q4f16`) | symmetric 4-bit, block 32: onnxruntime's `quantize_matmul_4bits` (fp16 input for `q4f16`) |
| `*_quant` + `*_scales` + `*_zp` (vision `q4`/`q4f16`/`quantized`, decoder `quantized`, all embeddings) | asymmetric 4- or 8-bit, block 32: onnxruntime's `quant_slice_asymmetric` (round half to even), computed in the scales' dtype |

Self-test: the same script, fed the reference's own fp32 weights instead of the Reasoning checkpoint, rebuilds the reference's fp16 and quantized variants. __SELFTEST__

The 4-bit differences come from the platform, not from the method: on Apple silicon `quantize_matmul_4bits` reproduces Mistral's bytes exactly (checked on `layers.0.q_proj`). The export here ran on x86-64, where onnxruntime's MLAS rounds the values that sit exactly on a rounding boundary the other way, one quantization step apart. The vision projector's `patch_merger` source weight in Mistral's quantized graphs differs slightly from the one in their own fp32 graph.

Tools: Python 3.12, torch 2.14.1 and torchvision 0.29.1 (CPU), transformers 5.20.1, onnx 1.23.2, onnxruntime 1.31.0, numpy 2.3.5, safetensors 0.8.0, huggingface_hub 2.2.0, on __BUILT_ON__. The export is the [`ministral-3-3b-reasoning-2512`](https://github.com/ai-ecoverse/models/tree/main/recipes/ministral-3-3b-reasoning-2512) recipe of [ai-ecoverse/models](https://github.com/ai-ecoverse/models); a copy of it is in [`export/`](export), with the pinned [`requirements.txt`](export/requirements.txt), the resolved [`environment.txt`](export/environment.txt) and [`provenance.json`](export/provenance.json) (commit and CI run). [`convert.sh`](export/convert.sh) runs, in order:

```sh
python lib/fetch.py mistralai/Ministral-3-3B-Instruct-2512-ONNX e1128d389145f96bb1c3c2625cb8ca255ffa7034 ref
python lib/fetch.py mistralai/Ministral-3-3B-Reasoning-2512 4a36357c811bf511a7b625d132e12f22408aac91 ckpt --exclude consolidated.safetensors
python selftest.py --ref ref --tmp selftest-tmp --out results/selftest.json --tolerance 0.0005
python build.py --ref ref --ckpt ckpt --out out --variants q4f16 q4 fp16 quantized fp32
python assemble.py --ref ref --ckpt ckpt --out out
python fix_llama4.py --out out --ckpt ckpt --results results
python parity.py torch --ckpt ckpt --repo out --ref-out results/torch
python parity.py onnx --repo out --variant q4f16 --ref-out results/torch --out results/onnx-q4f16.json   # and every other variant
python gate.py results q4f16 q4 fp16 quantized fp32
```

**Long-context fix.** Ministral 3 multiplies its queries by `1 + 0.1 · log(1 + floor(position / 16384))` (transformers' `get_llama_4_attn_scale`, with `original_max_position_embeddings` = 16384). Mistral's decoder graphs compute this in `/model/llama4_scaling` with 262144 as the divisor, and the fp16 graphs store that constant as fp16, where it overflows to inf, so the scale was 1 at every position and outputs drifted from PyTorch past position 16,383. `fix_llama4.py` replaces that subgraph in every decoder with one computed in float32 (fp16 can't represent positions above 2048 exactly) from the checkpoint's config, cast to the graph's dtype at the end, and checks it against the transformers formula at positions around each multiple of 16384. Revisions of this repo before the fix have the original graphs; the weights are unchanged.

`fetch.py` downloads with HTTP range resumption and checks every file's SHA-256 against the Hub before use. `gate.py` fails the build if any variant drifts further from PyTorch than this export's numbers allow. `build.py --variants fp32` rebuilds any variant from the base checkpoint in a few minutes on a CPU box with 32 GB of RAM. The upload was verified by downloading every file again and comparing SHA-256 with the local build and with the Hub's LFS metadata.
