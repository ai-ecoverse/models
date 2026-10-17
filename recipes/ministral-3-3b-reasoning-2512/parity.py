"""Greedy parity between the PyTorch checkpoint and the ONNX export.

    python parity.py torch --ckpt CKPT --repo OUT --ref-out results/torch        (PyTorch fp32 greedy + logits)
    python parity.py onnx --repo OUT --variant q4f16 --ref-out results/torch --out results/onnx-q4f16.json

The prompts are tokenized once (repo's own chat template and tokenizer via transformers) and the same ids go to both
sides. For every prompt the ONNX side reports the free-running greedy match against PyTorch (identical tokens before
the first divergence) and, teacher-forced on PyTorch's tokens, argmax agreement, mean KL(torch || onnx) and the
largest |logit difference|. Vision features are compared directly.
"""
import argparse, json, os, sys, time
import numpy as np

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
PROMPTS = [
    {"name": "think", "max_new": 1024, "messages": [
        {"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"}]},
    {"name": "tool", "max_new": 512, "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Use the get_weather tool to look up the current weather in Paris."}]},
    {"name": "terse", "max_new": 64, "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is the capital of France? One word."}]},
    {"name": "german", "max_new": 384, "messages": [
        {"role": "user", "content": "Erkläre in zwei Sätzen, warum der Himmel blau ist."}]},
    {"name": "image", "max_new": 256, "image": True, "messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]},
]
SUFFIX = {"fp32": "", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "quantized": "_quantized"}
IMAGE_TOKEN = 10
EOS = 2


def test_image():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 336), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 200, 160], fill=(30, 90, 200))
    d.ellipse([240, 60, 420, 240], fill=(220, 40, 40))
    d.polygon([(60, 320), (160, 190), (260, 320)], fill=(40, 170, 60))
    return im


def encode(repo, only=None, max_new=None):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(repo)
    out = []
    for p in PROMPTS:
        if only and p["name"] not in only:
            continue
        if max_new:
            p = dict(p, max_new=max_new)
        text = proc.apply_chat_template(p["messages"], tools=p.get("tools"), add_generation_prompt=True, tokenize=False)
        if p.get("image"):
            enc = proc(text=text, images=[test_image()], return_tensors="np", add_special_tokens=False)
            out.append(dict(p, ids=enc["input_ids"][0].astype(np.int64), pixel_values=enc["pixel_values"].astype(np.float32),
                            image_sizes=enc["image_sizes"], text=text))
        else:
            ids = proc.tokenizer(text, add_special_tokens=False, return_tensors="np")["input_ids"][0].astype(np.int64)
            out.append(dict(p, ids=ids, text=text))
    return proc, out


def run_torch(a):
    import torch
    from transformers import Mistral3ForConditionalGeneration
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    proc, prompts = encode(a.repo, a.only, a.max_new)
    model = Mistral3ForConditionalGeneration.from_pretrained(a.ckpt, dtype=torch.float32).to(a.device).eval()
    os.makedirs(a.ref_out, exist_ok=True)
    summary = {}
    for p in prompts:
        t0 = time.time()
        ids = torch.tensor(p["ids"])[None].to(a.device)
        kw = {}
        if p.get("image"):
            kw = dict(pixel_values=torch.tensor(p["pixel_values"]).to(a.device), image_sizes=torch.tensor(p["image_sizes"]).to(a.device))
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=p["max_new"],
                           eos_token_id=EOS, pad_token_id=11, **kw)
        new = g[0, ids.shape[1]:].cpu().numpy()
        full = g[:, :ids.shape[1] + len(new)]
        logits = model(input_ids=full, attention_mask=torch.ones_like(full), **kw).logits[0, ids.shape[1] - 1:-1].float().cpu().numpy()
        top2 = np.sort(logits, -1)[:, -2:]
        np.save(os.path.join(a.ref_out, f"{p['name']}.logits.npy"), logits.astype(np.float16))
        np.save(os.path.join(a.ref_out, f"{p['name']}.tokens.npy"), new)
        if p.get("image"):
            feats = model.get_image_features(pixel_values=kw["pixel_values"], image_sizes=kw["image_sizes"])
            feats = torch.cat(list(feats.pooler_output if hasattr(feats, "pooler_output") else feats), 0).float().cpu().numpy()
            np.save(os.path.join(a.ref_out, "image.features.npy"), feats)
        text = proc.tokenizer.decode(new, skip_special_tokens=False)
        summary[p["name"]] = {"prompt_tokens": int(len(p["ids"])), "new_tokens": int(len(new)), "text": text,
                              "min_margin": float((top2[:, 1] - top2[:, 0]).min()), "seconds": round(time.time() - t0, 1)}
        print(p["name"], summary[p["name"]]["new_tokens"], "tokens", f"{time.time() - t0:.0f}s", repr(text[:300]), flush=True)
    path = os.path.join(a.ref_out, "torch.json")
    old = json.load(open(path)) if os.path.exists(path) else {"prompts": {}, "torch": {}}
    old["prompts"].update({p["name"]: {"text": p["text"]} for p in prompts})
    old["torch"].update(summary)
    json.dump(old, open(path, "w"), indent=1, ensure_ascii=False)


class Onnx:
    def __init__(self, repo, variant, providers):
        import onnxruntime as ort
        s = SUFFIX[variant]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        mk = lambda n: ort.InferenceSession(os.path.join(repo, "onnx", n + s + ".onnx"), so, providers=providers)
        self.embed, self.dec = mk("embed_tokens"), mk("decoder_model_merged")
        self.vis = None
        self.mk = mk
        self.kv = {i.name: (np.float16 if i.type == "tensor(float16)" else np.float32) for i in self.dec.get_inputs() if i.name.startswith("past")}
        self.layers = len(self.kv) // 2

    def features(self, pixel_values):
        if self.vis is None:
            self.vis = self.mk("vision_encoder")
        return self.vis.run(None, {"pixel_values": pixel_values})[0]

    def embeds(self, ids, pixel_values=None):
        e = self.embed.run(None, {"input_ids": ids[None]})[0]
        if pixel_values is not None:
            f = self.features(pixel_values).reshape(-1, e.shape[-1])
            mask = ids == IMAGE_TOKEN
            if mask.sum() != f.shape[0]:
                raise SystemExit(f"{mask.sum()} image tokens but {f.shape[0]} features")
            e[0, mask] = f
        return e

    def step(self, embeds, past, start):
        n = embeds.shape[1]
        feed = {"inputs_embeds": embeds.astype(np.float32),
                "attention_mask": np.ones((1, start + n), np.int64),
                "position_ids": np.arange(start, start + n, dtype=np.int64)[None]}
        for i in range(self.layers):
            for kv in ("key", "value"):
                name = f"past_key_values.{i}.{kv}"
                feed[name] = past[name] if past else np.zeros((1, 8, 0, 128), self.kv[name])
        out = self.dec.run(None, feed)
        names = [o.name for o in self.dec.get_outputs()]
        logits = out[0]
        present = {n.replace("present", "past_key_values"): v for n, v in zip(names[1:], out[1:])}
        return logits, present

    def greedy(self, ids, max_new, pixel_values=None):
        logits, past = self.step(self.embeds(ids, pixel_values), None, 0)
        pos, new = len(ids), []
        while len(new) < max_new:
            t = int(np.argmax(logits[0, -1]))
            new.append(t)
            if t == EOS:
                break
            logits, past = self.step(self.embeds(np.array([t], np.int64)), past, pos)
            pos += 1
        return np.array(new, np.int64)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def run_onnx(a):
    proc, prompts = encode(a.repo, a.only, a.max_new)
    m = Onnx(a.repo, a.variant, a.providers.split(","))
    res = {"variant": a.variant, "providers": a.providers, "prompts": {}}
    for p in prompts:
        t0 = time.time()
        ref = np.load(os.path.join(a.ref_out, f"{p['name']}.tokens.npy"))
        ref_logits = np.load(os.path.join(a.ref_out, f"{p['name']}.logits.npy")).astype(np.float32)
        pv = p.get("pixel_values")
        new = m.greedy(p["ids"], p["max_new"], pv)
        n = min(len(new), len(ref))
        diverge = next((i for i in range(n) if new[i] != ref[i]), n if len(new) == len(ref) else n)
        full = np.concatenate([p["ids"], ref[:-1]]) if len(ref) else p["ids"]
        logits, _ = m.step(m.embeds(full, pv), None, 0)
        tf = logits[0, len(p["ids"]) - 1:]
        lt, lo = log_softmax(ref_logits), log_softmax(tf)
        kl = (np.exp(lt) * (lt - lo)).sum(-1)
        r = {"tokens": int(len(new)), "torch_tokens": int(len(ref)), "identical": bool(np.array_equal(new, ref)),
             "first_divergence": int(diverge), "tf_argmax_agree": float((tf.argmax(-1) == ref_logits.argmax(-1)).mean()),
             "tf_mean_kl": float(kl.mean()), "tf_max_kl": float(kl.max()), "tf_max_abs_logit": float(np.abs(tf - ref_logits).max()),
             "text": proc.tokenizer.decode(new, skip_special_tokens=False), "seconds": round(time.time() - t0, 1),
             "max_new": int(p["max_new"])}
        if pv is not None:
            ft = np.load(os.path.join(a.ref_out, "image.features.npy"))
            fo = m.features(pv).reshape(ft.shape)
            cos = (ft * fo).sum(-1) / np.linalg.norm(ft, axis=-1) / np.linalg.norm(fo, axis=-1)
            r["vision_max_abs"] = float(np.abs(ft - fo).max())
            r["vision_min_cos"] = float(cos.min())
        res["prompts"][p["name"]] = r
        print(p["name"], {k: v for k, v in r.items() if k != "text"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--repo", required=True)
    t.add_argument("--ref-out", required=True)
    t.add_argument("--device", default="cpu")
    t.add_argument("--only", nargs="*")
    t.add_argument("--max-new", type=int)
    o = sub.add_parser("onnx")
    o.add_argument("--repo", required=True)
    o.add_argument("--variant", required=True, choices=list(SUFFIX))
    o.add_argument("--ref-out", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--providers", default="CPUExecutionProvider")
    o.add_argument("--only", nargs="*")
    o.add_argument("--max-new", type=int)
    a = ap.parse_args()
    run_torch(a) if a.cmd == "torch" else run_onnx(a)


if __name__ == "__main__":
    sys.exit(main())
