"""Fail the build when a shipped variant drifts from PyTorch's `model.decide()`, or the reference from the card.

    python gate.py RESULTS --variants fp32 q8 [--experiments fp16 q8f16 q4 q4f16]

Over all 2,300 decisions (typed-decisions test, 2,000; transfer-v4, 300), on the calibrated probabilities. The
bounds follow kev.js: fp32 matches to 1e-4 with no answer changed; int8 (q8, fp32 activations) stays within a small
|Δp| and changes answers only on near-ties, where the reference itself is split. Measured on an M4 Max, 2026-10-28:
fp32 max 1.0e-5; q8 max 0.016, mean 0.0010, 9 of 2,300 answers changed, every one with a reference margin under
0.013. Experiments are recorded in gate.json and never fail the build.
"""
import argparse, json, os, sys

BOUNDS = {   # max |Δp|, mean |Δp|, changed answers allowed (fraction), reference margin below which a change is a tie
    "fp32": (1e-4, 1e-5, 0.0, 0.0),
    "q8": (0.04, 0.003, 0.01, 0.03),
}
# The source model on typed-decisions (all/test): model card 76.4 % against the gold label; HF's leaderboard (#37)
# 0.770 accuracy, KL 0.119, Brier 0.065.
CARD = {"accuracy_vs_label": (0.764, 0.002), "accuracy_tie_aware": (0.770, 0.002), "kl_from_gold": (0.119, 0.002),
        "brier": (0.065, 0.001)}


TIE = 0.03   # experiments: changes below this reference margin are reported as ties


def check(v, r):
    max_dp, mean_dp, frac, tie = BOUNDS.get(v, (None, None, None, TIE))
    sets = r["parity"]
    n = sum(p["calibrated"]["decisions"] for p in sets.values())
    mx = max(p["calibrated"]["max_abs_dp"] for p in sets.values())
    mean = sum(p["calibrated"]["mean_abs_dp"] * p["calibrated"]["decisions"] for p in sets.values()) / n
    changed = [c for p in sets.values() for c in p["calibrated"]["changed"]]
    clear = [c for c in changed if c["reference_margin"] >= tie]
    row = {"decisions": n, "max_abs_dp": mx, "mean_abs_dp": mean, "answers_changed": len(changed),
           "changed_outside_ties": len(clear), "max_changed_margin": max((c["reference_margin"] for c in changed), default=0),
           "texts_identical": not r["text_mismatches"], "tokens_identical": not r["token_mismatches"]}
    if max_dp is None:
        row["ok"] = None
        return row
    row["ok"] = (mx <= max_dp and mean <= mean_dp and len(changed) <= frac * n and not clear
                 and row["texts_identical"] and row["tokens_identical"])
    row["bounds"] = {"max_abs_dp": max_dp, "mean_abs_dp": mean_dp, "answers_changed": int(frac * n), "tie_margin": tie}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--experiments", nargs="*", default=[])
    a = ap.parse_args()
    ok = True
    ref = json.load(open(os.path.join(a.results, "torch.json")))
    td = ref["summary"]["typed-decisions"]["calibrated"]
    card = {k: {"measured": td[k], "card": c, "ok": abs(td[k] - c) <= tol} for k, (c, tol) in CARD.items()}
    for k, c in card.items():
        ok &= c["ok"]
        print(f"{'ok ' if c['ok'] else 'BAD'} PyTorch {k}: {c['measured']:.4f} (card {c['card']})")
    out = {"reference": card, "variants": {}, "experiments": {}}
    for v in a.variants + a.experiments:
        r = check(v, json.load(open(os.path.join(a.results, f"onnx-{v}.json"))))
        if v in a.variants:
            if v not in BOUNDS:
                raise SystemExit(f"no bounds for shipped variant {v}")
            ok &= r["ok"]
            out["variants"][v] = r
            tag = "ok " if r["ok"] else "BAD"
        else:
            out["experiments"][v] = r
            tag = "exp"
        print(f"{tag} {v}: max |Δp| {r['max_abs_dp']:.2g}, mean {r['mean_abs_dp']:.2g}, {r['answers_changed']}/"
              f"{r['decisions']} answers changed ({r['changed_outside_ties']} outside ties, largest margin "
              f"{r['max_changed_margin']:.3f}), texts {'=' if r['texts_identical'] else '≠'}, "
              f"tokens {'=' if r['tokens_identical'] else '≠'}")
    json.dump(out, open(os.path.join(a.results, "gate.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
