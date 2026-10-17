---
license: apache-2.0
base_model:
- __BASE__
base_model_relation: quantized
language:
- en
- fr
- es
- de
- it
- pt
- nl
- zh
- ja
- ko
- ar
library_name: transformers.js
pipeline_tag: image-text-to-text
tags:
- onnx
- onnxruntime
- mistral3
---

# __TITLE__

ONNX export of [__BASE__](https://huggingface.co/__BASE__) by [Mistral AI](https://mistral.ai), in the file layout of Mistral's official [__REFERENCE__](https://huggingface.co/__REFERENCE__): `onnx/decoder_model_merged_<variant>.onnx`, `onnx/embed_tokens_fp16.onnx` and `onnx/vision_encoder_fp16.onnx`, with weights in `.onnx_data` files of at most 2 GB each. Runtimes that load Mistral's 3B ONNX export (transformers.js, [slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama), plain onnxruntime) load this one, with the model's own tokenizer, chat template and generation config.

__ABOUT__

This is not an official Mistral AI release. All credit for the model goes to Mistral AI; the export was made by [ai-ecoverse](https://github.com/ai-ecoverse).

## Variants

__VARIANTS__

## Parity with PyTorch

__PARITY__

## Usage

Download a variant's files together with the JSON and template files in one `hf download`:

```sh
hf download __REPO__ --include '*.json' chat_template.jinja 'onnx/decoder_model_merged_q4f16.onnx*' 'onnx/embed_tokens_fp16.onnx*' 'onnx/vision_encoder_fp16.onnx*'
```

### Transformers.js

```js
import { AutoProcessor, AutoModelForImageTextToText, TextStreamer } from "@huggingface/transformers";

const model_id = "__REPO__";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModelForImageTextToText.from_pretrained(model_id, { device: "webgpu" });

const messages = [{ role: "user", content: "What is 17 multiplied by 23?" }];
const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
const inputs = await processor(prompt, null, { add_special_tokens: false });
await model.generate({
  ...inputs,
  max_new_tokens: 1024,
  do_sample: true,
  streamer: new TextStreamer(processor.tokenizer, { skip_prompt: true, skip_special_tokens: false }),
});
```

For images, pass `{ type: "image" }` in the message content and the image to the processor.

## How it was made

- **Decoder:** onnxruntime-genai's model builder (`Mistral3TextModel`, `exclude_embeds=true`, int4 `block_size=32`, `accuracy_level=4`; `-e webgpu` for `q4f16`, `-e cpu` for `q4`) from the base checkpoint at `__BASE_REVISION__` (bf16, __LAYERS__ layers, hidden size __HIDDEN__). Inputs are `inputs_embeds`, `attention_mask` and `past_key_values.N.{key,value}`; outputs `logits` and `present.N.{key,value}`. `inputs_embeds` and `logits` are float32 in every variant, as in Mistral's q4f16 decoder.
- **Long-context query scaling:** Ministral 3 multiplies its queries by `1 + 0.1 · log(1 + floor(position / 16384))` (transformers' `get_llama_4_attn_scale`). The builder doesn't emit this, so it is added to each layer's query projection, with positions taken from `attention_mask`. It changes nothing below position 16,384.
- **Vision encoder and embeddings:** every Ministral 3 size has the same Pixtral vision tower, so Mistral's own `vision_encoder_fp16.onnx` and `embed_tokens_fp16.onnx` graphs from [__REFERENCE__](https://huggingface.co/__REFERENCE__) at `__REFERENCE_REVISION__` are reused node for node, with this checkpoint's weights; only the projector and the embedding table take the larger hidden size. There are no 4-bit vision encoders: they lose too much (on Mistral's 3B export, worst-patch cosine 0.34 against fp32).
- **Configs:** `config.json` is the checkpoint's, with the text `model_type` transformers.js knows (`ministral`) and a `transformers.js_config` naming the default dtypes and data files. Tokenizer, chat template, processor and generation config are the checkpoint's; `preprocessor_config.json` is Mistral's from the 3B ONNX export.

Tools: __TOOLS__, on __BUILT_ON__. The export is the [`__RECIPE__`](https://github.com/ai-ecoverse/models/tree/main/recipes/__RECIPE__) recipe of [ai-ecoverse/models](https://github.com/ai-ecoverse/models); a copy is in [`export/`](export), with the resolved [`environment.txt`](export/environment.txt) and [`provenance.json`](export/provenance.json) (commit and CI run). Every file was verified after upload against the Hub's SHA-256 and a fresh download.

## License

Apache 2.0, as the base model: see the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt). The model is © Mistral AI.
