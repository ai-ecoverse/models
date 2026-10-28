"""Reference outputs (PyTorch), ONNX outputs per variant, and the metrics both sets are reported with.

    python evaluate.py torch --src SRC --td TEST.parquet --kev DEV.jsonl --out RESULTS/torch.json
    python evaluate.py onnx --export OUT [--onnx-dir DIR] --variant q8 --ref RESULTS/torch.json --out RESULTS/onnx-q8.json

`torch` runs the source's own `model.decide()` (fp32, CPU, trust_remote_code) on every request, and checks that
render.py rebuilds exactly the texts the source renders. `onnx` runs runtime.Decider (tokenizers + onnxruntime, no
PyTorch) on the same requests and compares texts, token ids and probabilities with the reference.

Metrics
  typed-decisions (the dataset card's bench/score_with_jobs.py, per decision over the 2,000): accuracy is argmax of
    the prediction against argmax of the gold distribution, KL(gold || pred) with pred floored at 1e-9, Brier is
    the sum over options of (pred - gold)^2, ECE over 10 confidence bins. `accuracy_vs_label` compares with the
    gold `label` field instead, as the model card's 76.4 % does; `accuracy_tie_aware` counts a prediction right
    when the gold distribution has a tie and the prediction picks any of the tied options, which gives the
    leaderboard's 0.770 (#37).
  transfer-v4 (kev.js evalset.py, per question over the 300): accuracy, and Brier as the sum over options of
    squared error against the one-hot label. `raw` is softmax(logits) (T = 1, as Kev's table), `calibrated` the
    model's per-type temperatures.
"""
import argparse, hashlib, json, math, os, platform, statistics, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, render  # noqa: E402

# The model card's example: three questions, one of each type (also the timing request).
README_REQUEST = {
    "state": {
        "agent": {"autonomy": "unsupervised", "model": "internal-agent-v2"},
        "task": "Clean up stale feature branches merged more than 90 days ago.",
        "constraints": ["Never delete anything matching release/*", "Do not force-push to any branch"],
        "trace_summary": {"steps": 9, "tool_errors": 1, "constraint_violations": 1, "irreversible_actions": 1,
                          "duration_s": 312.4},
    },
    "questions": {
        "action": {"type": "choice", "instructions": "What should the observability system do with this trace?",
                   "criteria": {"continue": "Let the agent proceed without interruption.",
                                "observe": "Keep running, but flag the trace for later sampling.",
                                "human_review": "Queue this trace for a human to review.",
                                "stop": "Halt the agent now."}},
        "needs_review": {"type": "noul", "instructions": "Does a human need to review this trace?"},
        "risk": {"type": "score", "instructions": "How risky is what the agent did?",
                 "criteria": ["no risk", "low risk", "significant risk", "severe risk"]},
    },
}
README_WORKFLOW = "agent_trace_observability"
TRANSFER_WORKFLOW = ""   # transfer-v4 records carry no workflow; decide()'s default


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()[:16]


def cases(td, kev):
    out = [("typed-decisions", c["name"], c["request"], c["workflow"], c) for c in data.typed_decisions(td)]
    out += [("transfer-v4", r["name"], r["request"], TRANSFER_WORKFLOW, r) for r in data.load_dev(kev)]
    return out


# ---- metrics ----------------------------------------------------------------------------------------------------
def td_score(specs, probs, gold):
    """Per decision, as bench/score_with_jobs.py (one request per case, every gold question)."""
    rows = []
    for spec, p in zip(specs, probs):
        gq = gold.get(spec["qid"])
        if gq is None:
            continue
        pred = dict(zip(spec["options"], p[: len(spec["options"])]))
        opts = list(gq["probabilities"])
        g = [float(gq["probabilities"][o]) for o in opts]
        q = [float(pred.get(o, 0.0)) for o in opts]
        s = sum(q)
        q = [v / s for v in q] if s > 0 else [1 / len(opts)] * len(opts)
        am = q.index(max(q))
        rows.append({"type": gq["type"], "ok": am == g.index(max(g)), "ok_tie": g[am] == max(g), "ok_label": opts[am] == str(gq["label"]).lower()
                     if gq["type"] == "noul" else opts[am] == str(gq["label"]),
                     "kl": sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, q) if gi > 0),
                     "brier": sum((pi - gi) ** 2 for pi, gi in zip(q, g)), "conf": max(q)})
    return rows


def td_summary(rows):
    n = len(rows)
    bins = [[0, 0, 0.0] for _ in range(10)]
    for r in rows:
        b = bins[min(int(r["conf"] * 10), 9)]
        b[0] += 1
        b[1] += r["ok"]
        b[2] += r["conf"]
    ece = sum(t / n * abs(h / t - c / t) for t, h, c in bins if t)
    out = {"decisions": n, "accuracy": sum(r["ok"] for r in rows) / n,
           "accuracy_vs_label": sum(r["ok_label"] for r in rows) / n,
           "accuracy_tie_aware": sum(r["ok_tie"] for r in rows) / n,
           "kl_from_gold": sum(r["kl"] for r in rows) / n, "brier": sum(r["brier"] for r in rows) / n, "ece": ece}
    for t in render.QUESTION_TYPES:
        sub = [r for r in rows if r["type"] == t]
        out[f"accuracy_{t}"] = sum(r["ok"] for r in sub) / len(sub)
        out[f"accuracy_vs_label_{t}"] = sum(r["ok_label"] for r in sub) / len(sub)
    return out


def tv_score(specs, probs, labels):
    rows = []
    for spec, p in zip(specs, probs):
        y = data.label_index(spec, labels[spec["qid"]])
        p = np.asarray(p[: len(spec["options"])], np.float64)
        rows.append({"ok": int(np.argmax(p) == y), "brier": float(((p - np.eye(len(p))[y]) ** 2).sum())})
    return rows


def tv_summary(rows, sources=None):
    out = {"questions": len(rows), "accuracy": sum(r["ok"] for r in rows) / len(rows),
           "brier": sum(r["brier"] for r in rows) / len(rows)}
    if sources:
        for s in sorted(set(sources)):
            sub = [r for r, x in zip(rows, sources) if x == s]
            out[f"accuracy_{s}"] = sum(r["ok"] for r in sub) / len(sub)
    return out


def summarize(records):
    """Metrics of a results file's records (reference or variant)."""
    td = [r for r in records if r["set"] == "typed-decisions"]
    tv = [r for r in records if r["set"] == "transfer-v4"]
    out = {}
    if td:
        out["typed-decisions"] = {
            "calibrated": td_summary([x for r in td for x in td_score(r["specs"], r["probabilities"], r["gold"])]),
            "raw": td_summary([x for r in td for x in td_score(r["specs"], r["raw_probabilities"], r["gold"])])}
    if tv:
        src = [r["source"] for r in tv for _ in r["specs"]]
        out["transfer-v4"] = {
            "raw": tv_summary([x for r in tv for x in tv_score(r["specs"], r["raw_probabilities"], r["labels"])], src),
            "calibrated": tv_summary([x for r in tv for x in tv_score(r["specs"], r["probabilities"], r["labels"])], src)}
    return out


def specs_slim(specs):
    return [{"qid": s["qid"], "kind": s["kind"], "options": s["options"]} for s in specs]


# ---- PyTorch reference --------------------------------------------------------------------------------------------
def run_torch(a):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.src, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.src, trust_remote_code=True, dtype=torch.float32).eval()
    src_mod = sys.modules[type(model).__module__]
    cap = {}

    def hook(mod, args, kwargs, out):
        cap["logits"], cap["ids"], cap["mask"] = out.logits, kwargs["input_ids"], kwargs["attention_mask"]
    model.register_forward_hook(hook, with_kwargs=True)

    records, render_bad = [], []
    t0 = time.time()
    for i, (set_name, name, req, wf, c) in enumerate(cases(a.td, a.kev)):
        ans = model.decide(req["state"], req["questions"], workflow=wf, tokenizer=tok)
        _, sp, texts = model._build_inputs([(wf, req["state"], req["questions"], None)], tok)
        mine, lay = render.build(req, wf)
        if mine != texts or [s["options"] for s in lay["specs"]] != [list(s.options) for s in sp]:
            render_bad.append(name)
        ids = [row[: int(m.sum())].tolist() for row, m in zip(cap["ids"], cap["mask"])]
        mask = torch.zeros(cap["logits"].shape, dtype=torch.bool)
        for q, s in enumerate(sp):
            mask[q, : len(s.options)] = True
        raw = cap["logits"].float().masked_fill(~mask, -1e4).softmax(-1)
        probs = [[ans[s.qid].probabilities[o] for o in s.options] for s in sp]
        rec = {"set": set_name, "name": name, "workflow": wf, "specs": specs_slim(lay["specs"]),
               "texts_sha": hashlib.sha256("\n\0".join(texts).encode()).hexdigest()[:16], "ids_sha": ids_digest(ids),
               "n_texts": len(texts), "max_tokens": max(len(x) for x in ids),
               "probabilities": probs, "raw_probabilities": [r[: len(s.options)].tolist() for r, s in zip(raw, sp)]}
        if set_name == "typed-decisions":
            rec["gold"] = c["gold"]
        else:
            rec["labels"], rec["source"] = c["labels"], c["source"]
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  torch {i + 1} requests, {time.time() - t0:.0f} s", flush=True)
    # render.py against the source's own functions on the hand-written edge cases too
    for fx in json.load(open(a.requests)):
        try:
            _, sp, texts = model._build_inputs([(fx.get("workflow", ""), fx["request"]["state"],
                                                 fx["request"]["questions"], None)], tok)
            ok = render.build(fx["request"], fx.get("workflow", ""))[0] == texts
        except Exception as e:   # noqa: BLE001
            try:
                render.build(fx["request"], fx.get("workflow", ""))
                ok = False
            except Exception as e2:   # noqa: BLE001
                ok = type(e) is type(e2)
        if not ok:
            render_bad.append(fx["name"])
    # timing of the card's 3-question request
    for _ in range(3):
        model.decide(README_REQUEST["state"], README_REQUEST["questions"], README_WORKFLOW, tokenizer=tok)
    ms = []
    for _ in range(a.repeat):
        t = time.perf_counter()
        model.decide(README_REQUEST["state"], README_REQUEST["questions"], README_WORKFLOW, tokenizer=tok)
        ms.append((time.perf_counter() - t) * 1e3)
    out = {"source": src_mod.__name__, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
           "render_mismatches": render_bad, "timing_ms": {"median": statistics.median(ms), "min": min(ms)},
           "records": records}
    out["summary"] = summarize(records)
    json.dump(out, open(a.out, "w"))
    print(json.dumps({"summary": out["summary"], "render_mismatches": render_bad[:5], "timing_ms": out["timing_ms"]},
                     indent=1), flush=True)
    if render_bad:
        raise SystemExit(f"render.py differs from the source on {len(render_bad)} requests: {render_bad[:5]}")


# ---- ONNX ---------------------------------------------------------------------------------------------------------
def parity(ref_recs, recs, key):
    d, changed, margins = [], [], []
    for r, g in zip(ref_recs, recs):
        for qi, (p, q) in enumerate(zip(r[key], g[key])):
            p, q = np.asarray(p), np.asarray(q)
            d.append(np.abs(p - q))
            srt = np.sort(p)[::-1]
            margin = float(srt[0] - srt[1]) if len(srt) > 1 else 1.0
            margins.append(margin)
            if int(np.argmax(p)) != int(np.argmax(q)):
                changed.append({"name": r["name"], "question": r["specs"][qi]["qid"], "reference_margin": margin})
    d = np.concatenate(d)
    return {"max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()), "answers_changed": len(changed),
            "decisions": len(margins), "changed": changed}


def run_onnx(a):
    import onnxruntime as ort
    from runtime import Decider
    ref = json.load(open(a.ref))
    t = time.perf_counter()
    rt = Decider(a.export, a.variant, a.onnx_dir)
    load_s = time.perf_counter() - t
    recs, text_bad, ids_bad = [], [], []
    t0 = time.time()
    for i, ((set_name, name, req, wf, c), r) in enumerate(zip(cases(a.td, a.kev), ref["records"])):
        assert r["name"] == name
        o = rt.decide(req, wf)
        if hashlib.sha256("\n\0".join(o["texts"]).encode()).hexdigest()[:16] != r["texts_sha"]:
            text_bad.append(name)
        if ids_digest(o["ids"]) != r["ids_sha"]:
            ids_bad.append(name)
        n = [len(s["options"]) for s in r["specs"]]
        recs.append({**{k: r[k] for k in ("set", "name", "specs") + (("gold",) if "gold" in r else ("labels", "source"))},
                     "probabilities": [p[:k].tolist() for p, k in zip(o["probabilities"], n)],
                     "raw_probabilities": [p[:k].tolist() for p, k in zip(o["raw_probabilities"], n)]})
        if (i + 1) % 100 == 0:
            print(f"  {a.variant} {i + 1} requests, {time.time() - t0:.0f} s", flush=True)
    for _ in range(3):
        rt.decide(README_REQUEST, README_WORKFLOW)
    ms = [rt.decide(README_REQUEST, README_WORKFLOW)["ms"]["total"] for _ in range(a.repeat)]
    by_set = {}
    for s in ("typed-decisions", "transfer-v4"):
        rr = [x for x in ref["records"] if x["set"] == s]
        gg = [x for x in recs if x["set"] == s]
        by_set[s] = {"calibrated": parity(rr, gg, "probabilities"), "raw": parity(rr, gg, "raw_probabilities")}
    out = {"variant": a.variant, "onnxruntime": ort.__version__,
           "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
           "text_mismatches": text_bad, "token_mismatches": ids_bad, "parity": by_set,
           "summary": summarize(recs), "load_s": load_s,
           "timing_ms": {"median": statistics.median(ms), "min": min(ms)}, "seconds": time.time() - t0}
    json.dump(out, open(a.out, "w"), indent=1)
    short = {s: {k: {m: v[m] for m in ("max_abs_dp", "mean_abs_dp", "answers_changed")} for k, v in p.items()}
             for s, p in by_set.items()}
    print(json.dumps({"variant": a.variant, "parity": short, "texts": len(text_bad), "tokens": len(ids_bad),
                      "timing_ms": out["timing_ms"]}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("torch")
    t.add_argument("--src", required=True)
    t.add_argument("--requests", required=True, help="fixtures/requests.json (hand-written edge cases)")
    o = sub.add_parser("onnx")
    o.add_argument("--export", required=True)
    o.add_argument("--onnx-dir")
    o.add_argument("--variant", required=True)
    o.add_argument("--ref", required=True)
    for p in (t, o):
        p.add_argument("--td", required=True)
        p.add_argument("--kev", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--repeat", type=int, default=20)
    a = ap.parse_args()
    return run_torch(a) if a.cmd == "torch" else run_onnx(a)


if __name__ == "__main__":
    sys.exit(main())
