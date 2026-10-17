"""Export a causal LM to ONNX (logits only, no KV cache), quantize it to int8, and gate both on logit parity.

    python export.py --src SRC --out OUT --results RESULTS
"""
import argparse, json, os, shutil, sys
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCausalLM

TOL = {"model.onnx": (1e-4, 1.0), "model_quantized.onnx": (0.5, 0.9)}   # max |Δlogit|, argmax agreement


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.float32, attn_implementation="eager").eval()
    os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)
    fp32 = os.path.join(a.out, "onnx", "model.onnx")
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        torch.onnx.export(Logits(model), (ids, mask), fp32, dynamo=False, opset_version=17,
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch", 1: "seq"}})
    quantize_dynamic(fp32, os.path.join(a.out, "onnx", "model_quantized.onnx"), weight_type=QuantType.QUInt8)

    test = torch.randint(0, model.config.vocab_size, (3, 37))
    with torch.no_grad():
        ref = model(input_ids=test, attention_mask=torch.ones_like(test)).logits.numpy()
    report, ok = {}, True
    for name, (max_abs, agree) in TOL.items():
        sess = ort.InferenceSession(os.path.join(a.out, "onnx", name), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input_ids": test.numpy(), "attention_mask": np.ones_like(test.numpy())})[0]
        r = {"max_abs_logit": float(np.abs(got - ref).max()), "argmax_agree": float((got.argmax(-1) == ref.argmax(-1)).mean())}
        r["ok"] = r["max_abs_logit"] <= max_abs and r["argmax_agree"] >= agree
        ok &= r["ok"]
        report[name] = r
        print(name, r, flush=True)
    os.makedirs(a.results, exist_ok=True)
    json.dump(report, open(os.path.join(a.results, "parity.json"), "w"), indent=1)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
    if not ok:
        raise SystemExit("parity gate failed")
    print("parity gate passed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
