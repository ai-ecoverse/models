"""Upstream kev.train's LoRA path on EmbeddingGemma 2, made resumable (upstream resumes full-weight runs only).

The loop is kev.train.main's, line for line, built from its own pieces (training_requests, microbatch_plan,
encode_batch, row_passes, batch_loss, Meta / finish_checkpoint); upstream's argument parser reads the same flags. Added:

- a resume point (trainable tensors, optimizer, schedule, counters, RNG states) every --save_every_steps optimizer
  steps and at the end of every epoch, written atomically to <run>/resume; a restart continues from it;
- a heartbeat (<run>/heartbeat.json: step, MPS current / driver bytes) for the watchdog after every optimizer step;
- records whose augmented variant no longer fits upstream's 1,024-token row limit under Gemma's tokenizer are skipped
  for that epoch and counted (upstream would abort: its suites were admitted under Qwen's tokenizer).

    python -m kev_eg2.train --run runs/<arm> -- --suite evals/v7/decision-v7 --epochs 2 --lr 1e-4 --lora 16 --batch 1 --accum 8 \
        --p_none_pair 0.25 --special_embeddings 1 --device mps
"""
import argparse, contextlib, json, math, os, random, resource, sys, time
from collections import Counter
from pathlib import Path
from . import adapt
import torch
import kev.train as kt
from kev.checkpoint import Meta
from kev.device import allocated_bytes, empty_cache, select, sync
from kev.model import ContextOverflow, load_tokenizer
from kev.suite import digest, read_manifest, write_json


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory: checkpoint/ (final, upstream layout), resume/, heartbeat.json")
    ap.add_argument("--save_every_steps", type=int, default=250)
    ap.add_argument("--max_rejected", type=int, default=50, help="fail if more augmented variants than this overflow the row limit")
    ap.add_argument("upstream", nargs=argparse.REMAINDER)
    ours = ap.parse_args(argv)
    up = ours.upstream[1:] if ours.upstream[:1] == ["--"] else ours.upstream
    run = Path(ours.run).resolve()
    if "--out" in up: raise SystemExit("--out is <run>/checkpoint")
    placeholder = run / f".parse-{os.getpid()}"   # upstream refuses an existing --out (and resume for LoRA runs): parse against a fresh path
    saved = sys.argv
    try:
        sys.argv = ["kev.train", *up, "--out", str(placeholder)]
        a = kt.parse_args()
    finally:
        sys.argv = saved
    a.out = str(run / "checkpoint")
    if a.full_ft or a.anchor or a.perm_kl or a.shared_prefix or a.row_budget or a.length_sort or a.init_from or a.data:
        raise SystemExit("only upstream's plain LoRA suite recipe is adapted here")
    if a.dtype == "bf16" or a.weights_dtype == "bf16": raise SystemExit("fp32 only on MPS")
    return ours, a, run


def encode_chunk(model, tok, a, chunk, ep, rejected):
    """kev.train.encode_batch per record; a record whose variant overflows the context is skipped this epoch."""
    out = []
    for req in chunk:
        try:
            out += kt.encode_batch(model, tok, a, [req], ep)
        except ContextOverflow as e:
            rejected.append({"epoch": ep, "id": req["_meta"]["id"], "source": req["_meta"]["source"], "error": str(e)})
    return out


def trainable(model):
    return {k: p for k, p in model.named_parameters() if p.requires_grad}


def save_resume(run, model, opt, sched, state):
    d = run / "resume"; d.mkdir(parents=True, exist_ok=True)
    tmp = d / "point.pt.tmp"
    torch.save({"trainable": {k: v.detach().cpu() for k, v in trainable(model).items()}, "opt": opt.state_dict(), "sched": sched.state_dict(),
                "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
                "state": state}, tmp)
    os.replace(tmp, d / "point.pt")
    print(f"resume point: epoch {state['epoch']} micro-batch {state['microbatch']} step {state['step']}", flush=True)


def heartbeat(run, dev, step, steps):
    hb = {"pid": os.getpid(), "time": time.time(), "step": step, "steps": steps}
    if dev == "mps": hb.update(mps_current=torch.mps.current_allocated_memory(), mps_driver=torch.mps.driver_allocated_memory())
    tmp = run / "heartbeat.json.tmp"; tmp.write_text(json.dumps(hb)); os.replace(tmp, run / "heartbeat.json")
    return hb


def main(argv=None):
    ours, a, run = parse(sys.argv[1:] if argv is None else argv)
    adapt.check_upstream(); os.chdir(adapt.KEV_ROOT)
    if (run / "checkpoint" / "head.pt").exists(): raise SystemExit(f"{run} is finished")
    run.mkdir(parents=True, exist_ok=True)
    dev = select(a.device)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    autocast = contextlib.nullcontext()
    manifest = read_manifest(a.suite) if a.suite else None
    revision = kt.pinned_revision(a, manifest)   # the suite does not pin this base: --base_revision does
    if revision != adapt.BASE_REVISION: raise SystemExit(f"base revision {revision} is not the pinned {adapt.BASE_REVISION}")
    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]
    tok = load_tokenizer(a.base, revision=revision)
    model = kt.DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                             special_embeddings=bool(a.special_embeddings), dtype=torch.float32)
    if not isinstance(model, adapt.EG2DecisionModel): raise SystemExit("kev.train.DecisionModel is not the adapted model")
    if a.checkpointing: model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    meta = Meta(base=a.base, base_revision=revision, lora=a.lora, head_dim=a.head_dim, option_isolation=False,
                special_embeddings=bool(a.special_embeddings), weights_dtype="fp32", holdout=holdout, weights="lora")
    print(f"device={dev} attention={model.attention} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M "
          f"delimiters {adapt.DELIMS} -> {model.delims}", flush=True)
    reqs = kt.training_requests(a, tok, manifest, holdout)
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    eg2 = {"attention": model.attention, "kev_commit": adapt.KEV_COMMIT, "base_revision": revision, "delimiters": adapt.DELIMS,
           "implementation": model.lm.config._attn_implementation, "save_every_steps": ours.save_every_steps}
    write_json(run / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "eg2": eg2, "holdout": holdout,
                                              "env": {k: v for k, v in os.environ.items() if k.startswith(("PYTORCH_MPS", "KEV_"))}})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in kt.materialize(r)['questions']))}", flush=True)

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = sum(ends for _, _, ends in kt.microbatch_plan(reqs, a, 1, 0))
    steps = a.epochs * per_epoch
    steps = min(steps, a.max_steps) if a.max_steps else steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    st = {"step": 0, "seen": 0, "tokens_seen": 0, "peak_mem": 0, "peak_driver": 0, "optimizer_seconds": 0.0, "step_seconds": [], "elapsed": 0.0,
          "epoch": 0, "microbatch": 0, "grad_norms": [], "run": {}, "rejected": [], "resumes": 0}
    point = run / "resume" / "point.pt"
    if point.exists():
        p = torch.load(point, map_location="cpu", weights_only=False)
        own = trainable(model)
        if set(p["trainable"]) != set(own): raise SystemExit("resume point does not match this model's trainable tensors")
        with torch.no_grad():
            for k, v in p["trainable"].items(): own[k].copy_(v)
        opt.load_state_dict(p["opt"]); sched.load_state_dict(p["sched"])
        torch.set_rng_state(p["torch_rng"])
        if p["mps_rng"] is not None and dev == "mps": torch.mps.set_rng_state(p["mps_rng"])
        st = p["state"]; st["resumes"] += 1
        print(f"resumed from {point}: step {st['step']}, epoch {st['epoch']}, micro-batch {st['microbatch']}", flush=True)
    step, seen, tokens_seen = st["step"], st["seen"], st["tokens_seen"]
    run_terms = Counter(st["run"]); grad_norms = st["grad_norms"]; rejected = st["rejected"]
    model.train(); t0, last = time.time() - st["elapsed"], time.time()
    step_at_start, t_start = step, time.time()

    def snapshot(ep, mb):
        st.update(step=step, seen=seen, tokens_seen=tokens_seen, elapsed=time.time() - t0, epoch=ep, microbatch=mb,
                  grad_norms=grad_norms, run=dict(run_terms), rejected=rejected)
        return st

    done = False
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        if ep < st["epoch"]: continue   # finished epochs' shuffles are replayed, so the interrupted epoch's order returns
        plan = kt.microbatch_plan(reqs, a, 1, 0)
        for mb in range(st["microbatch"] if ep == st["epoch"] else 0, len(plan)):
            chunk, step_records, ends_step = plan[mb]
            batch = encode_chunk(model, tok, a, chunk, ep, rejected)
            if len(rejected) > ours.max_rejected: raise SystemExit(f"{len(rejected)} records overflow the context: {rejected[-3:]}")
            if batch:
                variants = sum(v.share for v in batch)
                group_records = step_records * (variants / len(chunk))
                for part in kt.row_passes(batch, a.row_budget, a.shared_prefix):
                    loss, terms = kt.batch_loss(model, a, part, dev, {}, None, autocast)   # raises on a non-finite loss
                    (loss / group_records).backward()
                    run_terms += terms
                run_terms["n"] += variants; seen += round(variants); tokens_seen += sum(v.tokens for v in batch)
            st["peak_mem"] = max(st["peak_mem"], allocated_bytes(dev))
            if ends_step:
                norm = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), kt.MAX_GRAD_NORM, error_if_nonfinite=True))
                started = time.time(); opt.step(); sync(dev); st["optimizer_seconds"] += time.time() - started
                grad_norms += [[] for _ in range(ep + 1 - len(grad_norms))]
                grad_norms[ep].append(round(norm, 6))
                sched.step(); opt.zero_grad(); step += 1
                st["step_seconds"].append(round(time.time() - last, 3)); last = time.time()
                if dev == "mps": empty_cache(dev)
                hb = heartbeat(run, dev, step, steps)
                st["peak_driver"] = max(st["peak_driver"], hb.get("mps_driver", 0))
                if step % 10 == 0:
                    per_step = (time.time() - t_start) / max(1, step - step_at_start)
                    print(f"ep{ep} step {step}/{steps} loss {run_terms['ce']/max(run_terms['n'], 1e-9):.3f} {(time.time() - t0) / max(seen, 1):.3f}s/rec {per_step:.2f}s/step "
                          f"lr {sched.get_last_lr()[0]:.2e} gnorm {norm:.3f} mps {hb.get('mps_current', 0)/2**30:.2f}/{hb.get('mps_driver', 0)/2**30:.2f} GB "
                          f"eta {per_step * (steps - step) / 60:.0f} min", flush=True)
                    run_terms = Counter()
                if step == steps: done = True; break
                if ours.save_every_steps and step % ours.save_every_steps == 0:
                    save_resume(run, model, opt, sched, snapshot(ep, mb + 1)); last = time.time()
        if done: break
        save_resume(run, model, opt, sched, snapshot(ep + 1, 0)); last = time.time()   # every epoch

    wall = time.time() - t0
    model.lm.save_pretrained(a.out)
    backbone_seconds = time.time() - t0 - wall
    meta.head = model.head.state_dict()
    meta.extra = {"args": vars(a), "suite_sha256": suite_hash, "init_source": None, "eg2": eg2}
    kt.finish_checkpoint(a.out, meta, tok)
    write_json(Path(a.out) / "training_metrics.json", {"wall_seconds": wall, "records_seen": round(seen),
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": len(rejected), "rejected": rejected,
               "optimizer_steps": step, "forward_tokens": round(tokens_seen), "step_seconds": st["step_seconds"], "optimizer_seconds": st["optimizer_seconds"],
               "resume_seconds": [], "resume_write_seconds": [], "world_size": 1, "resumes": st["resumes"],
               "backbone_save_seconds": round(backbone_seconds, 1), "snapshots": [], "grad_norm": kt.grad_norm_summary(grad_norms),
               "weights": "lora", "peak_device_bytes": st["peak_mem"], "peak_driver_bytes": st["peak_driver"], "device": dev, "dtype": a.dtype, "batch": a.batch,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
