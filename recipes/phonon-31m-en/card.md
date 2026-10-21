---
license: apache-2.0
base_model:
- __SOURCE_REPO__
base_model_relation: quantized
language:
- en
library_name: onnx
pipeline_tag: text-to-speech
tags:
- onnx
- onnxruntime
- text-to-speech
- tts
- phonon
- slicc-ortllama
---

# Phonon 31m English ONNX

ONNX export of [__SOURCE_REPO__](https://huggingface.co/__SOURCE_REPO__) (revision `__SOURCE_REVISION__`), the 31M-parameter English text-to-speech model by [Gradium](https://huggingface.co/gradium), built on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) and the Mimi codec, with the __VOICE_COUNT__ English voices of [__VOICES_REPO__](https://huggingface.co/__VOICES_REPO__) (revision `__VOICES_REVISION__`). It runs in plain onnxruntime and through the `phonon` engine of [slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama) (`POST /v1/audio/speech`).

All credit for the model and the voices goes to Gradium. This repo only converts them.

## License

Apache-2.0, like the model ([__SOURCE_REPO__](https://huggingface.co/__SOURCE_REPO__)) and the voices ([__VOICES_REPO__](https://huggingface.co/__VOICES_REPO__)) it is made from, both published by Gradium under Apache-2.0.

## Files

| File | What |
| --- | --- |
| `config.json` | the checkpoint's own config, with an `onnx` section: graph shapes, the Mimi decoder's state tensors, the voices and the default voice (`__DEFAULT_VOICE__`) |
| `tokenizer.json` | the checkpoint's tokenizer (Unigram with byte fallback) |
| `onnx/text_conditioner.onnx` | token IDs to text embeddings, plus the first step's embedding |
| `onnx/flow_lm_main.onnx` | the flow LM transformer: embeddings and `past_key_values.*` to the last hidden state, the end-of-speech logit and `present.*` |
| `onnx/flow_lm_flow.onnx` | the flow head: hidden state and noise to one 32-dim Mimi latent, and the next step's embedding |
| `onnx/mimi_decoder.onnx` | the streaming Mimi decoder: N latents, an offset and __MIMI_STATE__ state tensors to 1920·N samples at 24 kHz and the next state |
| `voices/<name>.bin` | voice prompts, already projected for the flow LM: `[125, 512]` float32, little-endian |
| `export/` | the recipe that made this export, with its environment and provenance |

All graphs are opset 17. Variants:

| Variant | Suffix | What | Size |
| --- | --- | --- | --- |
__VARIANTS__

A graph without a variant file (the text conditioner and Mimi decoder of `q8` and `q4`) is used in fp32.

Voices (__VOICE_COUNT__): __VOICES__.

## Parity

Checked against Gradium's runtime, ptts __PTTS_VERSION__ (Rust, [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts) at `__PTTS_REVISION__`), in f32 on the CPU: __SENTENCES__ English sentences in the voice Easton, temperature 0.3, seed 42, __FRAMES__ frames, with ptts's noise replayed into onnxruntime.

Teacher-forced (each frame starts from ptts's previous latent; the Mimi decoder decodes ptts's latents):

| Variant | Latent max abs error | Relative error, median / p99 | End-of-speech logit max error | Mimi audio SNR, min / median |
| --- | --- | --- | --- | --- |
__PARITY__

Free-running `fp32` with ptts's noise: __FREE__.

Voices: every `voices/<name>.bin` matches ptts's projection of the voice __VOICE_MATCH__.

Text: slicc-ortllama __ORTLLAMA__'s English normalizer and tokenizer give ptts's normalized text, chunks and token IDs on __TOKENS__.

## Use

With [slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama) __ORTLLAMA__ or later, which reads models from the Hugging Face cache:

```bash
npm install -g @ai-ecoverse/slicc-ortllama onnxruntime-node
hf download ai-ecoverse/phonon-31m-en-ONNX
ortllama serve
curl -s http://127.0.0.1:11435/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "ai-ecoverse/phonon-31m-en-ONNX", "input": "Hello from Phonon.", "voice": "easton"}' -o hello.mp3
```

In plain onnxruntime: feed the voice prompt through `flow_lm_main` from an empty cache, then the text embeddings of each chunk, then one frame at a time: `flow_lm_main` on the previous step's embedding gives the hidden state and the end-of-speech logit, `flow_lm_flow` turns it and Gaussian noise (variance = temperature) into a latent and the next embedding, and `mimi_decoder` turns latents into audio. Stop a few frames after the end-of-speech logit passes `eos_threshold`.
