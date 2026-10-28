"""The transfer-v4 comparison with Kev, and the tables for the card.

    python compare.py RESULTS --kevjs KEVJS_FIXTURES_DIR --variants fp32 q8 --experiments fp16 ... --export OUT

Kev's per-record numbers come from kev.js's fixtures/kev-*-transfer-v4-dev300.json (raw logits, T = 1, its fp32
ONNX reference, which matches PyTorch to 1e-4): the same 300 records in the same order, so each question can be
compared. kev-ministral has no per-record file; its row is ortllama's README (0.733 / 0.414).
"""
import argparse, json, os, sys
import numpy as np

KEV = {   # kev.js fixture -> label, q8f32 download, served q8f32 accuracy / Brier (ortllama / kev.js README)
    "kev-0.8b": ("Kev-0.8B (`kev-0.8b@9a45d25`)", "0.84 GB", (0.640, 0.492)),
    "kev-4b": ("Kev-4B (`kev-4b@139fdd9`)", "4.7 GB", (0.800, 0.281)),
    "kev-9b": ("Kev-9B v2 (`kev-9b@b5d8c18`)", "8.8 GB", (0.807, 0.312)),
}
MINISTRAL = ("kev-ministral (`kev-ministral-3b`)", "6.6 GB", (0.733, 0.414))
SIZE = {"fp32": "onnx/model.onnx", "q8": "onnx/model_quantized.onnx"}


def kev_rows(path, names):
    d = json.load(open(path))
    recs = d["records"]
    if [r["name"] for r in recs] != names:
        raise SystemExit(f"{path}: not the same 300 records in the same order")
    ok, brier = [], []
    for r in recs:
        for p, y in zip(r["probs"], r["labels"]):
            p = np.asarray(p, np.float64)
            ok.append(int(np.argmax(p) == y))
            brier.append(float(((p - np.eye(len(p))[y]) ** 2).sum()))
    return d["run"], ok, brier


def mine(records, key):
    import evaluate as E
    rows = [x for r in records if r["set"] == "transfer-v4" for x in E.tv_score(r["specs"], r[key], r["labels"])]
    return [x["ok"] for x in rows], [x["brier"] for x in rows]


def size(export, graph):
    p = os.path.join(export, graph)
    return sum(os.path.getsize(f) for f in (p, p + "_data", os.path.join(export, "onnx", "decision_head.onnx")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--kevjs", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--experiments", nargs="*", default=[])
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ref = json.load(open(os.path.join(a.results, "torch.json")))
    tv = [r for r in ref["records"] if r["set"] == "transfer-v4"]
    names, sources = [r["name"] for r in tv], [r["source"] for r in tv]
    srcs = sorted(set(sources))

    def per_source(ok):
        return {s: sum(o for o, x in zip(ok, sources) if x == s) / sources.count(s) for s in srcs}

    td_ok, td_b = mine(ref["records"], "raw_probabilities")
    td_okc, td_bc = mine(ref["records"], "probabilities")
    out = {"records": len(tv), "sources": srcs, "rows": []}
    out["rows"].append({"model": "typed-decision, PyTorch fp32 (`model.decide()`)", "download": None,
                        "accuracy": float(np.mean(td_ok)), "brier": float(np.mean(td_b)),
                        "accuracy_calibrated": float(np.mean(td_okc)), "brier_calibrated": float(np.mean(td_bc)),
                        "per_source": per_source(td_ok)})
    for v in a.variants + a.experiments:
        p = os.path.join(a.results, f"onnx-{v}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))["summary"]["transfer-v4"]
        out["rows"].append({"model": f"typed-decision ONNX `{v}`" + (" (experiment)" if v in a.experiments else ""),
                            "download": f"{size(a.export, SIZE[v]) / 1e9:.2f} GB" if v in SIZE else None,
                            "accuracy": r["raw"]["accuracy"], "brier": r["raw"]["brier"],
                            "accuracy_calibrated": r["calibrated"]["accuracy"],
                            "brier_calibrated": r["calibrated"]["brier"],
                            "per_source": {s: r["raw"][f"accuracy_{s}"] for s in srcs}})
    agree = {}
    for key, (label, dl, readme) in KEV.items():
        path = os.path.join(a.kevjs, f"{key}-transfer-v4-dev300.json")
        run, ok, b = kev_rows(path, names)
        out["rows"].append({"model": label, "run": run, "download": dl, "accuracy": float(np.mean(ok)),
                            "brier": float(np.mean(b)), "ortllama_readme": readme, "per_source": per_source(ok)})
        both = sum(x and y for x, y in zip(td_ok, ok))
        agree[key] = {"both_right": both, "only_typed_decision": sum(td_ok) - both, f"only_{key}": sum(ok) - both,
                      "both_wrong": sum((not x) and (not y) for x, y in zip(td_ok, ok))}
    out["rows"].append({"model": MINISTRAL[0], "download": MINISTRAL[1], "accuracy": MINISTRAL[2][0],
                        "brier": MINISTRAL[2][1], "ortllama_readme": MINISTRAL[2], "per_source": None})
    out["agreement_raw"] = agree
    json.dump(out, open(os.path.join(a.results, "transfer-v4.json"), "w"), indent=1)

    lines = ["| Model | Download | Accuracy / Brier, T = 1 (reference) | Served | " + " | ".join(srcs) + " |",
             "|---|---|---|---|" + "---|" * len(srcs)]
    for r in out["rows"]:
        if "accuracy_calibrated" in r:
            served = f"{r['accuracy_calibrated']:.3f} / {r['brier_calibrated']:.3f} (calibrated)"
        else:
            served = f"{r['ortllama_readme'][0]:.3f} / {r['ortllama_readme'][1]:.3f} (`q8f32`, T = 1)"
        acc = "–" if r["per_source"] is None else f"{r['accuracy']:.3f} / {r['brier']:.3f}"
        ps = " | ".join(f"{r['per_source'][s]:.2f}" for s in srcs) if r["per_source"] else " | ".join("–" for _ in srcs)
        lines.append(f"| {r['model']} | {r['download'] or '–'} | {acc} | {served} | {ps} |")
    open(os.path.join(a.results, "transfer-v4.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(agree))


if __name__ == "__main__":
    sys.exit(main())
