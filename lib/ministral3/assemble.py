"""Put the non-ONNX files of a Ministral 3 export in place, in Mistral's transformers.js layout.

    python assemble.py --ref REF --ckpt CKPT --out OUT [--temperature 0.7 --top-p 0.95]

config.json is the checkpoint's, made to read like Mistral's Ministral-3-3B-Instruct-2512-ONNX config: no
quantization_config, the text model_type transformers.js knows ("ministral") with explicit layer_types, and a
transformers.js_config naming the default dtype per graph (q4f16 decoder, fp16 vision encoder and embeddings) and
each graph's number of external-data files. preprocessor_config.json comes from the reference (the vision
preprocessing is the same for every Ministral 3 size; checkpoints don't ship one). Tokenizer, chat template,
special tokens, processor and generation config come from the checkpoint; the generation config also gets the
sampling Mistral recommends for the model, when given. tokenizer_config.json gets the template inline, as in the
reference.
"""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shards import external_files  # noqa: E402

FROM_CKPT = ["tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json", "SYSTEM_PROMPT.txt"]
KV16 = {"q4f16", "fp16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    a = ap.parse_args()
    onnx_dir = os.path.join(a.out, "onnx")
    graphs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(onnx_dir, "*.onnx")))
    if not graphs:
        raise SystemExit(f"no graphs in {onnx_dir}")

    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    cfg.pop("quantization_config", None)
    tc = cfg["text_config"]
    tc["model_type"] = "ministral"
    tc["layer_types"] = ["full_attention"] * tc["num_hidden_layers"]
    # A separate lm_head in the checkpoint means untied embeddings; say so (the Reasoning configs leave it out,
    # and transformers' default would tie them).
    index = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    if "language_model.lm_head.weight" in index:
        cfg["tie_word_embeddings"] = tc["tie_word_embeddings"] = False
    variants = sorted(g[len("decoder_model_merged"):-len(".onnx")].lstrip("_") or "fp32" for g in graphs if g.startswith("decoder_model_merged"))
    cfg["transformers.js_config"] = {
        "dtype": {
            "decoder_model_merged": "q4f16" if "q4f16" in variants else variants[0],
            "vision_encoder": "fp16",
            "embed_tokens": "fp16",
        },
        "use_external_data_format": {g: len(external_files(onnx_dir, g)) for g in graphs},
        "kv_cache_dtype": {v: "float16" for v in variants if v in KV16},
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    shutil.copyfile(os.path.join(a.ref, "preprocessor_config.json"), os.path.join(a.out, "preprocessor_config.json"))
    for f in FROM_CKPT:
        if os.path.exists(os.path.join(a.ckpt, f)):
            shutil.copyfile(os.path.join(a.ckpt, f), os.path.join(a.out, f))
    gc = json.load(open(os.path.join(a.ckpt, "generation_config.json")))
    if a.temperature is not None:
        gc["temperature"] = a.temperature
    if a.top_p is not None:
        gc["top_p"] = a.top_p
    json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
    tok = json.load(open(os.path.join(a.ckpt, "tokenizer_config.json")))
    tok["chat_template"] = open(os.path.join(a.ckpt, "chat_template.jinja")).read()
    json.dump(tok, open(os.path.join(a.out, "tokenizer_config.json"), "w"), indent=2, ensure_ascii=False)
    print("assembled", sorted(os.listdir(a.out)), json.dumps(cfg["transformers.js_config"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
