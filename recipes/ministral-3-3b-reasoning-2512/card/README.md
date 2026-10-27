---
license: apache-2.0
base_model:
- mistralai/Ministral-3-3B-Reasoning-2512
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
- reasoning
---

# Ministral 3 3B Reasoning 2512 ONNX

ONNX export of [mistralai/Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512), the reasoning post-trained 3.4B language model + 0.4B Pixtral vision encoder by [Mistral AI](https://mistral.ai), in exactly the file layout of Mistral's official [mistralai/Ministral-3-3B-Instruct-2512-ONNX](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX). Anything that runs the Instruct ONNX export (transformers.js, [slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama), plain onnxruntime) runs this one unchanged, with the Reasoning model's own tokenizer, chat template and generation config.

The model thinks before it answers: the chat template's default system prompt asks for a `[THINK]…[/THINK]` block, followed by the answer. A request that brings its own system prompt replaces that default; Mistral recommends appending [`SYSTEM_PROMPT.txt`](SYSTEM_PROMPT.txt) to custom system prompts and keeping reasoning traces in multi-turn context. `generation_config.json` carries Mistral's recommended sampling for this model, `temperature` 0.7 and `top_p` 0.95, so runtimes that read it (transformers.js, slicc-ortllama) sample that way unless a request says otherwise. See the [base model card](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) for capabilities, benchmarks and intended use.

This is not an official Mistral AI release. All credit for the model goes to Mistral AI; the export was made by [ai-ecoverse](https://github.com/ai-ecoverse).

## Variants

__VARIANTS__

Each variant is three graphs, `onnx/decoder_model_merged{,_<variant>}.onnx`, `onnx/embed_tokens…` and `onnx/vision_encoder…`, with their weights in `.onnx_data` files of exactly the sizes Mistral's files have. transformers.js picks `q4f16` for the decoder by default (`transformers.js_config` in `config.json`).

## Parity with PyTorch

__PARITY__

## Usage

Download all the files of a variant (and the JSON and template files) together, in one `hf download`. A partial download made after the repo moved to a new revision points `refs/main` at a snapshot that holds only the newly fetched files, and a machine with an older, complete snapshot then can't resolve the model until it downloads the variant again.

### slicc-ortllama

[slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama) serves this model as its `chat-reasoning` role, with OpenAI-style `/v1/chat/completions` that stream `reasoning_content` and tool calls:

```sh
hf download ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX --include '*.json' chat_template.jinja 'onnx/*_q4f16.onnx*'
ortllama serve
curl -s localhost:11435/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "chat-reasoning",
  "messages": [{ "role": "user", "content": "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. What does the ball cost?" }]
}'
```

### Transformers.js

```js
import { AutoProcessor, AutoModelForImageTextToText, TextStreamer } from "@huggingface/transformers";

const model_id = "ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModelForImageTextToText.from_pretrained(model_id, { device: "webgpu" });

const messages = [{ role: "user", content: "What is 17 multiplied by 23?" }];
const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
const inputs = await processor(prompt, null, { add_special_tokens: false });
const output = await model.generate({
  ...inputs,
  max_new_tokens: 2048,
  do_sample: true,
  streamer: new TextStreamer(processor.tokenizer, { skip_prompt: true, skip_special_tokens: false }),
});
```

`[THINK]` and `[/THINK]` are special tokens; keep `skip_special_tokens: false` to see where the reasoning ends. For images, pass `{ type: "image" }` in the message content and the image to the processor, as on the [Instruct ONNX card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX).

### Parsing replies

`tokenizer_config.json` carries a [response template](https://huggingface.co/docs/transformers/en/chat_response_parsing), so transformers' `parse_response` (and `get_response_parser` for streaming) turns a raw reply into an assistant message with `reasoning_content`, `content` and `tool_calls`, ready to append to the conversation:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX")
prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
reply = tokenizer.decode(output_ids, skip_special_tokens=False)  # keep [THINK], [TOOL_CALLS], [ARGS] and </s>
message = tokenizer.parse_response(reply, prefix=prompt, tools=tools)
# {"role": "assistant", "reasoning_content": "…", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}
```

The reply must be decoded with its special tokens: without them the reasoning, the tool calls and the answer run together. Tool calls are `[TOOL_CALLS]name[ARGS]{…}`, several in a row for parallel calls, and the turn ends with `</s>`. slicc-ortllama parses replies the same way with its own parser.

## How it was made

__PROVENANCE__

## License

Apache 2.0, as the base model: see the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt). Ministral 3 3B Reasoning 2512 is © Mistral AI. The ONNX graphs are Mistral AI's own, from [mistralai/Ministral-3-3B-Instruct-2512-ONNX](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX) (Apache 2.0, contributed by Xenova of Hugging Face), with the Reasoning checkpoint's weights in place of the Instruct weights.
