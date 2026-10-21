---
license: cc-by-4.0
language:
- en
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: __SOURCE_REPO__
base_model_relation: quantized
tags:
- onnx
- parakeet
- tdt
- speech-to-text
- asr
- slicc-ortllama
---

# Underdog Bark 3.0, ONNX

[Underdog Bark 3.0](https://huggingface.co/__SOURCE_REPO__) by Conway Research, an English speech recognition model
fine-tuned from NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), exported to
ONNX for [slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama) and plain onnxruntime. It was converted from
`model.bark` at revision `__SOURCE_REVISION__`. That file is unpacked exactly with the source repo's own
`bark_loader.py`, then exported with `torch.onnx.export`.

## Files

| Variant | Encoder | decoder_joint | Size |
| --- | --- | --- | --: |
__VARIANT_ROWS__

A variant is the encoder and decoder_joint graphs that share a suffix. The decoder_joint graph (prediction network
and joint, ~70 MB) is fp32 in every variant.

## Graphs

`encoder_model*.onnx`:
- Inputs: `input_features` f32 `[B, T, 128]` and `attention_mask` i64 `[B, T]`.
- Outputs:
  - `encoder_outputs` f32 `[B, T', 640]`, the FastConformer output after `encoder_projector`, 8x subsampled;
  - `encoder_lengths` i64 `[B]`.

`decoder_joint_model*.onnx` computes one greedy TDT step:
- Inputs:
  - `encoder_outputs` f32 `[B, 1, 640]`, one encoder frame;
  - `input_ids` i64 `[B, 1]`, the last non-blank token (blank `8192` before the first);
  - `state_h` and `state_c` f32 `[2, B, 640]`.
- Outputs:
  - `token_logits` f32 `[B, 8193]`;
  - `duration_logits` f32 `[B, 5]`, over `durations` `[0, 1, 2, 3, 4]`;
  - `state_h_out` and `state_c_out`.

Greedy decoding, the way transformers' `ParakeetForTDT.generate()` does it:

1. Start at `t = 0` with `last = 8192` and zero states.
2. At frame `t`, take `tok = argmax(token_logits)` and `dur = durations[argmax(duration_logits)]`.
3. If `tok` is blank, advance by `max(dur, 1)` and leave the state as it is.
4. Otherwise, emit `tok`, set `last = tok`, take `state_*_out`, and advance by `dur` (0 stays on the frame).
5. Stop when `t >= encoder_lengths`.

Text is the emitted SentencePiece pieces joined, with `▁` turned into a space. Consecutive repeats are not merged.

## Features

The log-mel features are computed outside the graphs. `preprocessor_config.json` lists every parameter of
transformers' `ParakeetFeatureExtractor`:
- 0.25 s of leading silence;
- pre-emphasis 0.97;
- centred STFT, n_fft 512, hop 160, symmetric Hann(400);
- power spectrum into 128 Slaney mel filters;
- `log(x + 2^-24)`;
- per-feature mean and unbiased std over the valid frames, `floor(samples / 160)`, with `1e-5` added to the std;
- padded frames set to 0.

[`export/tdt.py`](export/tdt.py) is a NumPy reference client. `fixture/` holds one clip, its features from
transformers, the mel filters, the encoder output, and the expected tokens, durations and text for every parity
clip and variant.

## Parity

The reference is transformers' feature extractor plus `generate()` on the unpacked fp32 weights. It was run on __CLIPS__
English clips made with macOS `say`. The export was run on the same features, decoded greedily by `tdt.py`, with word
error rate as slicc-ortllama computes it (case and punctuation folded). fp32 on `tdt.py`'s own NumPy features also
matches token for token.

| | Exact tokens and durations | WER vs PyTorch | WER vs script |
| --- | --: | --: | --: |
__PARITY_ROWS__

## License

CC-BY-4.0, the same as the source; see `NOTICE`. The model derives from:
- Underdog Bark 3.0 by Conway Research;
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA;
- [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) by Moondream.

The conversion recipe is in `export/`, from [ai-ecoverse/models](https://github.com/ai-ecoverse/models).
