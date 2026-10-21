import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUFFIX = {"fp32": "", "q8": "_quantized", "q4": "_q4", "fp16": "_fp16"}

GATES = {
    "fp32": {"latent_max": 2e-4, "latent_rel_p50": 1e-5, "eos_max": 2e-4, "eos_flips": 0, "mimi_snr_min": 100.0},
    "fp16": {"latent_rel_p50": 2e-3, "eos_max": 0.1, "mimi_snr_min": 45.0},
    "q8": {"latent_rel_p50": 3e-2, "eos_max": 0.5},
    "q4": {"latent_rel_p50": 0.25, "eos_max": 3.0},
}
FREE_SNR_MIN = 20.0
VOICE_MAX = 1e-5
DELTAS = (-0.3, 0.3)


def snr(ref, out):
    return float(10 * np.log10((ref**2).sum() / max(((out - ref) ** 2).sum(), 1e-20)))


def eos_policy(frames_after_eos):
    countdown = [None]

    def step(eos):
        if eos and countdown[0] is None:
            countdown[0] = frames_after_eos
        if countdown[0] is None:
            return False
        if countdown[0] == 0:
            return True
        countdown[0] -= 1
        return False

    return step


class Pipeline:
    def __init__(self, root, variant, threads, delta=0.0):
        self.cfg = json.load(open(os.path.join(root, "config.json")))
        o = self.cfg["onnx"]
        self.layers, self.heads, self.head_dim = o["num_layers"], o["num_heads"], o["head_dim"]
        self.state = o["mimi_state"]
        self.threshold = self.cfg["eos_threshold"]
        self.delta = np.array([delta], np.float32)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.files = {}

        def session(name):
            f = os.path.join(root, "onnx", f"{name}{SUFFIX[variant]}.onnx")
            if not os.path.exists(f):
                f = os.path.join(root, "onnx", f"{name}.onnx")
            self.files[name] = os.path.relpath(f, root)
            return ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])

        self.text = session("text_conditioner")
        self.main = session("flow_lm_main")
        self.flow = session("flow_lm_flow")
        self.mimi = session("mimi_decoder")

    def conditioned(self, session, feeds):
        if "duration_delta" in [i.name for i in session.get_inputs()]:
            feeds = {**feeds, "duration_delta": self.delta}
        return session.run(None, feeds)

    def step(self, hidden, noise):
        return self.conditioned(self.flow, {"hidden": hidden, "noise": np.array([noise], np.float32)})

    def empty(self):
        return [np.zeros((1, self.heads, 0, self.head_dim), np.float32) for _ in range(2 * self.layers)]

    def run_main(self, embeds, past):
        feeds = {"inputs_embeds": embeds}
        for i in range(self.layers):
            feeds[f"past_key_values.{i}.key"] = past[2 * i]
            feeds[f"past_key_values.{i}.value"] = past[2 * i + 1]
        out = self.main.run(None, feeds)
        return out[0], float(out[1][0, 0]), out[2:]

    def prompt(self, voice, tokens):
        _, _, past = self.run_main(voice, self.empty())
        emb, bos = self.conditioned(self.text, {"input_ids": np.array([tokens], np.int64)})
        _, _, past = self.run_main(emb, past)
        return past, bos

    def decoder(self):
        st = {f"state.{n}": np.zeros(s, np.float32) for n, s in self.state}
        offset = [np.array(0, np.int64)]

        def decode(latent):
            out = self.mimi.run(None, {"latent": latent, "offset": offset[0], **st})
            offset[0] = out[1]
            for (n, _), v in zip(self.state, out[2:]):
                st[f"state.{n}"] = v
            return out[0][0, 0]

        return decode


def reference(ref_dir):
    docs = []
    for f in sorted(x for x in os.listdir(ref_dir) if x.endswith(".json")):
        doc = json.load(open(os.path.join(ref_dir, f)))
        doc["pcm"] = np.fromfile(os.path.join(ref_dir, f.replace(".json", ".f32")), np.float32)
        doc["file"] = f
        docs.append(doc)
    return docs


def delta(docs):
    values = {float(dict(c.split("=", 1) for c in d.get("conditions", [])).get("duration_delta", 0)) for d in docs}
    if len(values) != 1:
        raise SystemExit(f"reference sentences disagree on duration_delta: {sorted(values)}")
    return values.pop()


def teacher_forced(p, docs, voice, w_in):
    lat_err, rel, eos_err, mimi = [], [], [], []
    flips = 0
    for doc in docs:
        pcm = []
        for c in doc["chunks"]:
            past, x = p.prompt(voice, c["tokens"])
            decode = p.decoder()
            for fr in c["frames"]:
                hidden, e, past = p.run_main(x, past)
                latent, nxt = p.step(hidden, fr["noise"])
                r = np.array(fr["latent"], np.float32)
                lat_err.append(float(np.abs(latent[0, 0] - r).max()))
                rel.append(float(np.linalg.norm(latent[0, 0] - r) / max(np.linalg.norm(r), 1e-9)))
                eos_err.append(abs(e - fr["eos"]))
                flips += int((e > p.threshold) != (fr["eos"] > p.threshold))
                x = (nxt[0, 0] + (r - latent[0, 0]) @ w_in.T).astype(np.float32)[None, None, :]
                pcm.append(decode(r[None, None, :]))
        mimi.append(snr(doc["pcm"], np.concatenate(pcm)))
    lat_err, rel = np.array(lat_err), np.array(rel)
    return {
        "frames": int(len(lat_err)),
        "latent_max": float(lat_err.max()),
        "latent_p50": float(np.median(lat_err)),
        "latent_rel_p50": float(np.median(rel)),
        "latent_rel_p99": float(np.percentile(rel, 99)),
        "eos_max": float(max(eos_err)),
        "eos_flips": flips,
        "mimi_snr_min": float(min(mimi)),
        "mimi_snr_median": float(np.median(mimi)),
    }


def free_running(p, docs, voice):
    rows = []
    for doc in docs:
        pcm, frames, ref_frames = [], 0, 0
        for c in doc["chunks"]:
            past, x = p.prompt(voice, c["tokens"])
            decode = p.decoder()
            stop = eos_policy(c["frames_after_eos"])
            n = 0
            for f in range(c["frame_budget"]):
                noise = c["frames"][f]["noise"] if f < len(c["frames"]) else [0.0] * len(c["frames"][0]["noise"])
                hidden, e, past = p.run_main(x, past)
                latent, x = p.step(hidden, noise)
                pcm.append(decode(latent))
                n += 1
                if stop(e > p.threshold):
                    break
            frames += n
            ref_frames += len(c["frames"])
        out = np.concatenate(pcm)
        same = frames == ref_frames
        rows.append(
            {
                "file": doc["file"],
                "frames": frames,
                "ref_frames": ref_frames,
                "snr": snr(doc["pcm"], out) if same else None,
            }
        )
    snrs = [r["snr"] for r in rows if r["snr"] is not None]
    return {
        "sentences": len(rows),
        "same_length": sum(r["frames"] == r["ref_frames"] for r in rows),
        "pcm_snr_min": float(min(snrs)) if snrs else None,
        "pcm_snr_median": float(np.median(snrs)) if snrs else None,
        "rows": rows,
    }


def voices(out_dir, ref_dir):
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    rows = {}
    for v in cfg["onnx"]["voices"]:
        ours = np.fromfile(os.path.join(out_dir, "voices", f"{v['name']}.bin"), np.float32)
        ref = np.fromfile(os.path.join(ref_dir, f"{v['voice_id']}.f32"), np.float32)
        rows[v["name"]] = float(np.abs(ours - ref).max()) if ours.shape == ref.shape else None
    return rows


def gate_run(tf, free, at):
    problems = []
    for variant, limits in GATES.items():
        got = tf.get(variant)
        if got is None:
            problems.append(f"{variant}{at}: not measured")
            continue
        for key, limit in limits.items():
            ok = got[key] >= limit if key.startswith("mimi_snr") else got[key] <= limit
            if not ok:
                problems.append(f"{variant}{at}: {key} {got[key]:.3g} (limit {limit:g})")
    if free is None:
        return problems + [f"fp32 free-running{at}: not measured"]
    if free["same_length"] != free["sentences"]:
        problems.append(f"fp32 free-running{at}: {free['sentences'] - free['same_length']} sentence(s) stop on another frame than ptts")
    if free["pcm_snr_min"] is not None and free["pcm_snr_min"] < FREE_SNR_MIN:
        problems.append(f"fp32 free-running{at}: PCM SNR {free['pcm_snr_min']:.1f} dB (limit {FREE_SNR_MIN:g})")
    return problems


def gate(results):
    problems = gate_run(results.get("teacher_forced", {}), results.get("free_running"), "")
    for d in DELTAS:
        got = results["duration_delta"].get(str(d))
        if got is None:
            problems.append(f"duration_delta={d}: not measured")
            continue
        problems += gate_run(got["teacher_forced"], got["free_running"], f" at duration_delta={d}")
    bad = {k: v for k, v in results["voices"].items() if v is None or v > VOICE_MAX}
    if len(results["voices"]) != 37:
        problems.append(f"{len(results['voices'])} English voices, expected 37")
    if bad:
        problems.append(f"voices differ from ptts's projection: {bad}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True, action="append")
    ap.add_argument("--ref-voices", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    from export import Phonon

    ph = Phonon(a.model)
    w_in = ph.flow.liw.detach().numpy()
    results = {"graphs": {}, "duration_delta": {}}
    for ref in a.ref:
        docs = reference(ref)
        d = delta(docs)
        voice = np.fromfile(os.path.join(ref, "voice_emb.f32"), np.float32).reshape(1, -1, w_in.shape[0])
        run = results if d == 0 else results["duration_delta"].setdefault(str(d), {})
        run["teacher_forced"] = {}
        for variant in SUFFIX:
            p = Pipeline(a.out, variant, a.threads, d)
            results["graphs"][variant] = p.files
            r = teacher_forced(p, docs, voice, w_in)
            run["teacher_forced"][variant] = r
            print(f"duration_delta={d}", variant, json.dumps(r), flush=True)
        run["free_running"] = free_running(Pipeline(a.out, "fp32", a.threads, d), docs, voice)
        free = {k: v for k, v in run["free_running"].items() if k != "rows"}
        print(f"duration_delta={d}", "fp32 free-running", json.dumps(free), flush=True)
    results["voices"] = voices(a.out, a.ref_voices)
    worst = max((v for v in results["voices"].values() if v is not None), default=None)
    print(f"voices: {len(results['voices'])}, worst max abs {worst}", flush=True)
    results["problems"] = gate(results)
    results["ok"] = not results["problems"]
    with open(os.path.join(a.results, "parity.json"), "w") as f:
        json.dump(results, f, indent=1)
    for p in results["problems"]:
        print("FAIL", p)
    print("parity", "passed" if results["ok"] else "failed")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
