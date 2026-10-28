"""Score a trained arm exactly as upstream's studies score a trial (kev.experiment.score_trial: raw logits, a temperature
fitted on decision-v7 calibration, decision-v7 development, the mechanism checks, transfer-v4 development), then add
kev.js's 300-record transfer subset and per-source accuracy. Development partitions only: kev.suite.load_split refuses
the locked test without allow_test, which nothing here passes.

    python -m kev_eg2.score arm --run runs/bidirectional           # -> runs/bidirectional/trial/result.json, summary.json
    python -m kev_eg2.score summary --arms runs/bidirectional runs/causal --out runs/summary.json
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
from . import adapt
import torch
from kev.checkpoint import read_meta
from kev.experiment import score_trial, source_hashes
from kev.metrics import grouped_metrics, metrics
from kev.suite import digest, read_json, write_json

DECISION, TRANSFER = "evals/v7/decision-v7", "evals/v4/transfer-v4"
SELECTION_SHA256 = "89ae12319fd3bff17e6d9880297d008329bcc7d38fbfd19f17e67f6944f68dc7"   # kev.js load_dev(n=300, seed=1), as PR #25 pins it
KEVJS = Path(os.environ.get("KEVJS_ROOT", os.path.expanduser("~/Developer/ai-ecoverse/kev.js")))
KEYS = ("n", "acc", "brier", "nll", "ece")
# the spec's per-source groups: transfer-v4's sources; held-out policy = composition + contrastive
GROUPS = {"qnli": "QNLI", "sciq": "SciQ", "tweet_offensive": "TweetEval", "paws": "PAWS", "mmlu": "MMLU", "emotion": "Emotion",
          "composition_holdout": "policy", "contrastive": "policy", "legacy_holdout": "policy"}


def load_dev_ids(n=300, seed=1):
    """kev.js export/kev_web_export/fixtures.py load_dev, verbatim selection: shuffle with Random(seed), round-robin over sources."""
    rows = [json.loads(l) for l in open(Path(adapt.KEV_ROOT) / TRANSFER / "development.jsonl")]
    random.Random(seed).shuffle(rows)
    by_src, out = {}, []
    for r in rows:
        by_src.setdefault(r["_meta"]["source"], []).append(r)
    while len(out) < n and any(by_src.values()):
        for src in list(by_src):
            if by_src[src] and len(out) < n: out.append(by_src[src].pop())
    ids = [r["_meta"]["id"] for r in out]
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != SELECTION_SHA256: raise SystemExit("300-record selection differs from kev.js's")
    return ids


def pick(m):
    return {k: m[k] for k in KEYS if k in m}


def by_group(rows, T):
    groups = {}
    for r in rows: groups.setdefault(GROUPS.get(r["source"], r["source"]), []).append(r)
    return {g: {"n": len(rs), "acc": metrics(rs)["acc"], "brier_raw": metrics(rs)["brier"], "brier_cal": metrics(rs, T)["brier"]} for g, rs in sorted(groups.items())}


def extras(trial):
    """dev300 (all questions of kev.js's 300 records, as its evalset.py scores them) and per-source transfer metrics."""
    res = read_json(trial / "result.json"); T = res["temperature"]
    rows = read_json(trial / "transfer" / "rows.json")
    ids = set(load_dev_ids()); sub = [r for r in rows if r["id"] in ids]
    if len({r["id"] for r in sub}) != 300: raise SystemExit("dev300 records missing from the transfer rows")
    clean = [r for r in rows if r["variant"] == "clean"]
    return {"temperature": T,
            "transfer_v4_dev": {"raw": pick(res["transfer"]["clean"]), "calibrated": pick(res["transfer"]["calibrated_clean"])},
            "transfer_v4_dev300": {"questions": len(sub), "raw": pick(metrics(sub)), "calibrated": pick(metrics(sub, T))},
            "decision_v7_dev": {"raw": pick(res["clean"]), "calibrated": pick(res["calibrated_clean"]), "objective": res["objective"]},
            "transfer_by_source_clean": by_group(clean, T), "transfer_tasks_clean": {k: v["acc"] for k, v in res["transfer"]["tasks"].items()},
            "transfer_dev300_by_source": by_group(sub, T),
            "mechanism_checks": res["mechanism_checks"], "gates": res["gates"]["checks"],
            "training": {k: res["training_resources"].get(k) for k in ("wall_seconds", "optimizer_steps", "records_seen", "rejected_records", "peak_device_bytes", "peak_driver_bytes", "resumes")},
            "scoring_wall_seconds": res["wall_seconds"]}


def score_arm(run):
    run = Path(run).resolve(); ckpt = run / "checkpoint"; trial = run / "trial"
    meta = read_meta(ckpt)
    mode = meta.extra["eg2"]["attention"]
    if adapt.ATTENTION != mode: raise SystemExit(f"KEV_EG2_ATTN={adapt.ATTENTION} but {ckpt} was trained with {mode}")
    adapt.check_upstream(); os.chdir(adapt.KEV_ROOT)
    if not (trial / "result.json").exists():
        if trial.exists():   # an interrupted scoring: partial evaluation output only (upstream's resume_trial does the same)
            import shutil; shutil.rmtree(trial); print(f"removed partial evaluation {trial}", flush=True)
        trial.mkdir(parents=True)
        prov = {"config": meta.extra["args"], "suite_sha256": digest(Path(DECISION) / "manifest.json"), "source_hashes": source_hashes(),
                "kev_commit": adapt.KEV_COMMIT, "adaptation": "kev_eg2", "attention": mode, "torch": torch.__version__, "device": "mps"}
        write_json(trial / "provenance.json", prov)
        score_trial(str(ckpt), DECISION, trial, prov["source_hashes"], "mps", prov, TRANSFER, time.perf_counter(), legacy=False)
    out = extras(trial); out["attention"] = mode
    write_json(run / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("temperature", "transfer_v4_dev", "transfer_v4_dev300", "decision_v7_dev")}, indent=1))


def reference_rows():
    """Upstream's own trial results at the pinned commit (the same score_trial), and kev.js's dev300 fixtures."""
    ref = {}
    for name, path in (("Kev-0.8B stage 1 (q35-08b, seed 0)", "runs/q35-08b/00-trial-0"), ("Kev-0.8B stage 1 (q35-08b, seed 1)", "runs/q35-08b/01-trial-1"),
                       ("Kev-0.8B stage 1 (q35-08b, seed 2)", "runs/q35-08b/02-trial-2"),
                       ("Kev-0.6B (v7-06b, seed 0)", "runs/v7-06b/00-trial-0"), ("Kev-0.6B (v7-06b, seed 1)", "runs/v7-06b/01-trial-1"), ("Kev-0.6B (v7-06b, seed 2, released)", "runs/v7-06b/02-trial-2"),
                       ("Kev-0.8B shipped (r15-08b)", "runs/r15-08b/00-trial-0"), ("Kev-4B shipped (r10-skills)", "runs/r10-skills/00-trial-0")):
        p = Path(adapt.KEV_ROOT) / path / "result.json"
        if not p.exists(): continue
        r = read_json(p); t = r.get("transfer") or {}
        ref[name] = {"source": f"jaredpalmer/kev@{adapt.KEV_COMMIT[:7]}:{path}/result.json", "temperature": r.get("temperature"),
                     "transfer_v4_dev": {"raw": pick(t["clean"]), "calibrated": pick(t["calibrated_clean"])} if t else None,
                     "decision_v7_dev": {"raw": pick(r["clean"]), "calibrated": pick(r["calibrated_clean"])},
                     "transfer_tasks_clean": {k: v["acc"] for k, v in t.get("tasks", {}).items()}}
    dev = {json.loads(l)["_meta"]["id"]: json.loads(l)["_meta"]["source"] for l in open(Path(adapt.KEV_ROOT) / TRANSFER / "development.jsonl")}
    for size in ("0.8b", "4b", "9b"):
        p = KEVJS / "fixtures" / f"kev-{size}-transfer-v4-dev300.json"
        if not p.exists(): continue
        d = json.load(open(p)); per = {}
        for rec in d["records"]:
            g = GROUPS.get(dev[rec["name"]], dev[rec["name"]])
            for prob, y in zip(rec["probs"], rec["labels"]):
                per.setdefault(g, []).append(int(max(range(len(prob)), key=prob.__getitem__) == y))
        ref[f"Kev-{size.upper()} dev300 (kev.js fixture)"] = {"source": f"ai-ecoverse/kev.js fixtures/kev-{size}-transfer-v4-dev300.json ({d['run']}, shipped T)",
                                                             "transfer_v4_dev300": d["summary"], "by_source": {g: sum(v) / len(v) for g, v in sorted(per.items())}}
    return ref


def summary(arms, out):
    res = {"arms": {}, "references": reference_rows()}
    for a in arms:
        p = Path(a) / "summary.json"
        if p.exists(): res["arms"][Path(a).name] = read_json(p)
    # the arm is picked on decision-v7 development (upstream's study objective: -macro dev NLL), never on transfer
    if res["arms"]:
        chosen = max(res["arms"], key=lambda k: res["arms"][k]["decision_v7_dev"]["objective"])
        acc = res["arms"][chosen]["transfer_v4_dev"]["raw"]["acc"]
        res["selection"] = {"rule": "highest decision-v7 development objective (upstream's -macro NLL)", "chosen": chosen,
                            "transfer_v4_dev_acc": acc, "verdict": "pass" if acc >= 0.62 else "stop" if acc < 0.60 else "inconclusive"}
    write_json(out, res); print(json.dumps(res.get("selection"), indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arm"); s.add_argument("--run", required=True)
    s = sub.add_parser("summary"); s.add_argument("--arms", nargs="+", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "arm":
        from .watchdog import start_heartbeat
        (Path(a.run) / "score").mkdir(parents=True, exist_ok=True)
        start_heartbeat(str((Path(a.run) / "score" / "heartbeat.json").resolve()))
        score_arm(a.run)
    else:
        arms = [str(Path(x).resolve()) for x in a.arms]; out = str(Path(a.out).resolve())
        os.chdir(adapt.KEV_ROOT); summary(arms, out)


if __name__ == "__main__":
    main()
