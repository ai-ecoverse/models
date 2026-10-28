"""Fill card.md from the results.

    python card.py --template card.md --out OUT --results RESULTS --variants fp32 q8 --experiments fp16 ...
"""
import argparse, json, os

SUFFIX = {"fp32": "", "q8": "_quantized", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "q8f16": "_q8f16"}
BOUND = {"fp32": "≤ 1e-4, none changed", "q8": "≤ 0.04, mean ≤ 0.003, changes only below a 0.03 margin"}


def parity_row(v, r, g, fx, label=None):
    p = r["parity"]
    td, tv = p["typed-decisions"]["calibrated"], p["transfer-v4"]["calibrated"]
    cells = [label or f"`{v}`",
             f"{td['max_abs_dp']:.2g} / {td['mean_abs_dp']:.2g}", f"{td['answers_changed']} / {td['decisions']}",
             f"{tv['max_abs_dp']:.2g} / {tv['mean_abs_dp']:.2g}", f"{tv['answers_changed']} / {tv['decisions']}",
             f"{g['max_changed_margin']:.3f}" if g["answers_changed"] else "–",
             f"{fx['min_cosine']:.5f}" if fx else "–"]
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--experiments", nargs="*", default=[])
    a = ap.parse_args()
    e = os.environ
    R = lambda f: json.load(open(os.path.join(a.results, f)))   # noqa: E731
    torch_ref, fx = R("torch.json"), R("fixtures.json")
    import gate
    head = ["| Variant | typed-decisions max / mean \\|Δp\\| | answers changed | transfer-v4 max / mean \\|Δp\\| "
            "| answers changed | largest margin changed | min embedding cosine (fixtures) |", "|---" * 7 + "|"]
    par, exp = list(head), list(head)
    for v in a.variants:
        r = R(f"onnx-{v}.json")
        par.append(parity_row(v, r, gate.check(v, r), fx.get(v)))
    par.append("")
    par.append("Gate: " + "; ".join(f"`{v}` max |Δp| {BOUND[v]}" for v in a.variants if v in BOUND) + ".")
    for v in a.experiments:
        r = R(f"onnx-{v}.json")
        exp.append(parity_row(v, r, gate.check(v, r), fx.get(v)))
    if not a.experiments:
        exp = ["(none)"]

    rows = ["| Model | Accuracy | vs label | tie-aware | KL from gold | Brier | ECE | choice / score / noul |",
            "|---|---|---|---|---|---|---|---|"]

    def td_row(label, s):
        return (f"| {label} | {s['accuracy']:.4f} | {s['accuracy_vs_label']:.4f} | {s['accuracy_tie_aware']:.4f} | "
                f"{s['kl_from_gold']:.4f} | {s['brier']:.4f} | {s['ece']:.4f} | "
                f"{s['accuracy_choice']:.3f} / {s['accuracy_score']:.3f} / {s['accuracy_noul']:.3f} |")
    rows.append(td_row("PyTorch fp32 `decide()`", torch_ref["summary"]["typed-decisions"]["calibrated"]))
    for v in a.variants + a.experiments:
        rows.append(td_row(f"ONNX `{v}`" + (" (experiment)" if v in a.experiments else ""),
                           R(f"onnx-{v}.json")["summary"]["typed-decisions"]["calibrated"]))
    rows.append("| model card | – | 0.764 | – | – | 0.065 | – | 0.712 / 0.740 / 0.848 (vs label) |")
    rows.append("| leaderboard (#37) | – | – | 0.770 | 0.119 | 0.065 | 0.155 | – |")

    sizes = ["| Variant | Encoder | + head |", "|---|---|---|"]
    hb = os.path.getsize(os.path.join(a.out, "onnx", "decision_head.onnx"))
    for v in a.variants:
        p = os.path.join(a.out, "onnx", f"model{SUFFIX[v]}.onnx")
        n = os.path.getsize(p) + os.path.getsize(p + "_data")
        sizes.append(f"| `{v}` | {n / 1e6:,.0f} MB | {(n + hb) / 1e6:,.0f} MB |")

    built = [f"| this build ({R(f'onnx-{v}.json')['machine']}, onnxruntime {R(f'onnx-{v}.json')['onnxruntime']} "
             f"Python, CPU) | `{v}` | {R(f'onnx-{v}.json')['timing_ms']['median']:.0f} ms |" for v in a.variants]
    built.append(f"| this build, PyTorch {torch_ref['torch']} `decide()` | fp32 | {torch_ref['timing_ms']['median']:.0f} ms |")

    cfg = json.load(open(os.path.join(a.out, "decision_config.json")))
    temps = ", ".join(f"{t} {x:.4f}" for t, x in zip(cfg["question_types"], cfg["temperatures"]))
    nfx = len(json.load(open(os.path.join(a.out, "fixtures", "typed-decision.json")))["fixtures"])
    speed = open(os.path.join(os.path.dirname(os.path.abspath(a.template)), "card-speed.md")).read().strip()
    card = (open(a.template).read()
            .replace("__SOURCE_REPO__", e["RECIPE_SOURCE_REPO"]).replace("__SOURCE_REV__", e["RECIPE_SOURCE_REVISION"][:7])
            .replace("__FIXTURES__", str(nfx)).replace("__SIZES__", "\n".join(sizes)).replace("__TEMPS__", temps)
            .replace("__TORCH__", torch_ref["torch"])
            .replace("__DECISIONS__", f"{sum(p['calibrated']['decisions'] for p in R('onnx-fp32.json')['parity'].values()):,}")
            .replace("__PARITY__", "\n".join(par)).replace("__EXPERIMENTS__", "\n".join(exp))
            .replace("__TD__", "\n".join(rows))
            .replace("__TRANSFER__", open(os.path.join(a.results, "transfer-v4.md")).read().strip())
            .replace("__SPEED__", speed + "\n" + "\n".join(built)))
    open(os.path.join(a.out, "README.md"), "w").write(card)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
