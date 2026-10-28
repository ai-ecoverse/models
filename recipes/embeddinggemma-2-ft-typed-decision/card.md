---
license: apache-2.0
library_name: onnx
base_model: __SOURCE_REPO__
datasets:
- LocalLLaMA/typed-decisions
tags:
- onnx
- typed-decisions
- system-one
- embeddinggemma
---

# EmbeddingGemma 2 typed decisions, ONNX

[__SOURCE_REPO__](https://huggingface.co/__SOURCE_REPO__) at `__SOURCE_REV__` (Apache-2.0) in ONNX: a System One
decision model that answers `choice`, `score` and `noul` questions about a state with a probability per option. Built
by the [`embeddinggemma-2-ft-typed-decision`](https://github.com/ai-ecoverse/models/tree/main/recipes/embeddinggemma-2-ft-typed-decision)
recipe (`export/` holds the copy that made this folder), which checks every variant against the source's own
`model.decide()` in PyTorch.

Nothing here runs the model's Python. A runtime needs onnxruntime, `tokenizer.json` and a port of the text rendering,
which `fixtures/typed-decision.json` pins down stage by stage.

## Files

| File | What |
|---|---|
| `onnx/model.onnx` + `_data` | fp32 text encoder: [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)'s own text graph, unchanged, with the fine-tuned weights |
| `onnx/model_quantized.onnx` + `_data` | `q8`: int8 weights (8-bit blocks of 32, asymmetric), fp32 activations, quantized exactly as onnx-community quantized theirs (kev.js calls this `q8f32`) |
| `onnx/decision_head.onnx` | the decision head (~1.1M parameters) with the per-type temperatures, fp32, shared by every variant |
| `decision_config.json` | prompt, pooling, temperatures, tokenizer settings, graph inputs and outputs |
| `fixtures/typed-decision.json` | __FIXTURES__ requests with the source's texts, token ids, embeddings, logits, probabilities and answers |
| `config.json`, `tokenizer.json`, `tokenizer_config.json` | the source's |

__SIZES__

## How it runs

1. **Render** the request into unique texts, in order of first appearance: the state, then for each question its text
   and its options' texts. The rendering is part of the model (see below).
2. **Tokenize** `"task: classification | query: " + text` with `tokenizer.json`: `<bos> … <eos>`, at most 8,192
   tokens.
3. **Encode** with `onnx/model{suffix}.onnx`: `input_ids` and `attention_mask` (int64, right-padded), and empty
   `[0, 512]` float tensors for `image_features`, `video_features` and `audio_features`. Take `sentence_embedding`.
   It is the source's pooling exactly: the masked mean of `last_hidden_state` over every token, prompt included, then
   L2 normalization (`pooling_include_prompt: true`, no projections, all 768 dimensions); the two agree to 1.5e-7.
   Texts of similar length can share a batch; padding changes nothing beyond float rounding (2e-7).
4. **Score** with `onnx/decision_head.onnx`: the embeddings and `state_index`, `question_index`, `option_index`,
   `option_mask`, `question_type` (0 choice, 1 score, 2 noul). `probabilities` is calibrated (softmax of the logits
   over the type's temperature, __TEMPS__); `logits` gives T = 1.
5. **Answer**: the first option with the highest probability; `noul` → `label == "true"`, `score` → `int(label)` and
   the expected level.

### Rendering

From the source's `modeling_gemmadecision.py` (`parse_question`, `render_state`, `render_question`,
`render_option`); the recipe's `render.py` is a port that spells out Python's formatting.

- **State.** A dict becomes `workflow: {workflow}` and one `key: value` line per leaf, nested keys joined with ` · `
  and `_` replaced by spaces in the key path (`trace summary · tool errors: 1`). Any other state (string, list,
  number) is `workflow: {workflow} | state: {value}`. A string state that starts with `{` or `[` is parsed as JSON
  first. Values are formatted as Python's `str()`: `True`, `False`, `None`, `1.0`, `1e-07`, `1e+16`, lists and dicts as
  Python literals (`['a', "it's"]`, `{'k': 1}`).
- **Questions.** `{kind} question: {instructions}`; missing instructions fall back to the question id, a missing or
  unknown type to `choice`.
- **Options.** `question: {instructions} | answer: {answer}`, where the answer is a `choice` key (or `key: description`
  when they differ), `level {i} of {n-1}: {description}` for a `score`, and `no` / `yes` for a `noul` (options
  `false`, `true`; its criteria are ignored). `score` criteria may be a list, a dict with integer keys (sorted), or a
  dict of named levels (`name: description`, in order).
- `workflow` is part of the state text. The model was trained with the dataset's four workflows; the transfer numbers
  below use `""`.

## Parity

Against the source's `model.decide()` (PyTorch __TORCH__, fp32, CPU) on __DECISIONS__ decisions: LocalLLaMA/typed-decisions
`all`/`test` (400 requests, 2,000 decisions) and Kev's transfer-v4 300. Calibrated probabilities; a near-tie is a
reference margin (top two options) below the variant's tie bound.

__PARITY__

Rendering and token ids matched the source on every request. Experiments are built and measured but not shipped:

__EXPERIMENTS__

## typed-decisions (`all`, `test`)

The dataset card's scorer (`bench/score_with_jobs.py`): accuracy against the gold distribution's argmax, KL from gold,
Brier (sum over options), ECE. The model card's 76.4 % compares with the gold `label` instead; the leaderboard's 0.770
counts ties in the gold distribution as right for any tied option.

__TD__

## Kev's transfer-v4 (held-out, 300 questions)

kev.js's 300 held-out transfer-v4 development records (`load_dev(suite="evals/v4/transfer-v4", n=300, seed=1)`), the
original System One requests and labels, `workflow=""`. Accuracy and Brier (sum over options against the one-hot
label) as kev.js's `evalset.py`. Kev's columns are its fp32 reference from kev.js's fixtures; "served" is the
published `q8f32` bundle (ortllama's and kev.js's READMEs).

__TRANSFER__

The model was trained on four workflows (agent traces, customer service, invoices, security alerts) and does not
transfer to Kev's held-out tasks: at or below Kev-0.8B on every source (level on `tweet_offensive`), and below chance
on MMLU.

## Speed

One three-question request (the model card's example: 14 texts, 479 tokens, warm, median of 20; the recipe's
`bench.mjs` for the JavaScript runtimes). Every text is encoded; with the questions' and options' embeddings cached
(same schema as a previous request), only the state is, 24 ms (`fp32`) on the M4 Max's CPU.

__SPEED__

## Limitations

- **Rendering drift fails silently.** A port that formats a value differently (`1` for `1.0`, `true` for `True`,
  JSON for a list) gets other embeddings and other answers without an error; check it against the fixtures.
  JavaScript's `JSON.parse` loses `1.0` and orders integer-like object keys first.
- **fp16.** The model card warns that EmbeddingGemma 2 overflows to NaN in PyTorch fp16. onnx-community's fp16 graph
  with these weights did not, on the CPU or on WebGPU (the card's request), but it is not shipped.
- **4-bit** moves probabilities too far for a calibrated model (see the experiments).
- **WASM** is about 6× slower than the CPU on onnxruntime-node. `q8` (`GatherBlockQuantized` for the token embedding)
  ran on onnxruntime-web 1.30.0's WASM backend in Node; slicc-ortllama's embedding engine refuses quantized
  EmbeddingGemma 2 variants on WASM, so check it there before relying on it.
