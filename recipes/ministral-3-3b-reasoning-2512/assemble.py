"""Put the non-ONNX files of the export in place.

    python assemble.py --ref REF --ckpt CKPT --out OUT

From the reference ONNX repo: config.json (it carries transformers.js_config and the text model_type transformers.js
knows) and preprocessor_config.json (the base checkpoint has none). From the base checkpoint: its own tokenizer,
template, special tokens, processor and generation config; the generation config also gets the sampling Mistral's model
card recommends (temperature 0.7, top_p 0.95). tokenizer_config.json also gets the template inline, as in the
reference, and the response template in response_template.json (how transformers' parse_response turns a raw reply
into reasoning_content, content and tool_calls; contributed by Nico Martin of Hugging Face).
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROM_REF = ["config.json", "preprocessor_config.json"]
SAMPLING = {"temperature": 0.7, "top_p": 0.95}
FROM_CKPT = ["tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json", "generation_config.json", "SYSTEM_PROMPT.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref_cfg = json.load(open(os.path.join(a.ref, "config.json")))
    ck_cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    strip = lambda c: {k: v for k, v in c.items() if k not in ("model_type", "layer_types")}
    for key in ("text_config", "vision_config"):
        if strip(ref_cfg[key]) != strip(ck_cfg[key]):
            raise SystemExit(f"{key} differs between the reference and the checkpoint")
    for f in FROM_REF:
        shutil.copyfile(os.path.join(a.ref, f), os.path.join(a.out, f))
    for f in FROM_CKPT:
        shutil.copyfile(os.path.join(a.ckpt, f), os.path.join(a.out, f))
    gc = json.load(open(os.path.join(a.ckpt, "generation_config.json")))
    gc.update(SAMPLING)
    json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
    tc = json.load(open(os.path.join(a.ckpt, "tokenizer_config.json")))
    tc["chat_template"] = open(os.path.join(a.ckpt, "chat_template.jinja")).read()
    tc["response_template"] = json.load(open(os.path.join(HERE, "response_template.json")))
    json.dump(tc, open(os.path.join(a.out, "tokenizer_config.json"), "w"), indent=2, ensure_ascii=False)
    print("assembled", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    sys.exit(main())
