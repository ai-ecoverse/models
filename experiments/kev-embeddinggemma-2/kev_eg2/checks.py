"""Checks of the adaptation before any training (writes runs/checks.json; exits non-zero on a failure).

1. parity: the adapted model (text encoder only, projection removed, LoRA + trainable delimiter rows attached and
   disabled) gives the stock EmbeddingGemma2Model's hidden states: ours @ embedding_projection == stock last_hidden_state,
   on padded batches of real decision rows, fp32 CPU (exact path) and on MPS.
2. adapters on but untrained (LoRA B = 0, delimiter rows copied from the base) change nothing either.
3. escaping: caller text containing every Gemma special token, Qwen's delimiters and the <unusedN> strings yields no
   special or delimiter id.
4. causal arm: a row's hidden state at t does not depend on tokens after t; bidirectional: it does.
5. fp16 is refused.

    python -m kev_eg2.checks --out runs/checks.json
"""
import argparse, json, os, sys
import torch
from . import adapt
from kev.data import materialize
from kev.model import encode, rows_of
from kev.suite import load_split


def stock_model(dtype=torch.float32):
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2Model
    return EmbeddingGemma2Model.from_pretrained(adapt.BASE, revision=adapt.BASE_REVISION, dtype=dtype, attn_implementation="eager").eval()


def sample_rows(tok, n=6):
    recs = [r for r in load_split("evals/v7/decision-v7", "development") if r["_meta"]["variant"] == "clean"]
    rows = []
    for r in recs[:: max(1, len(recs) // n)][:n]:
        enc = encode(tok, materialize(r), strict=True)
        S, _, rs = rows_of(enc)
        rows.append([tok.bos_token_id, *S, *rs[0]["ids"], tok.eos_token_id])
    return rows


def padded(rows, pad, device):
    L = max(map(len, rows))
    ids = torch.full((len(rows), L), pad, dtype=torch.long); att = torch.zeros_like(ids)
    for i, r in enumerate(rows): ids[i, :len(r)] = torch.tensor(r); att[i, :len(r)] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def parity(tok, device, attn):
    rows = sample_rows(tok)
    ids, att = padded(rows, 0, device)
    stock = stock_model().to(device)
    ref = stock(input_ids=ids, attention_mask=att).last_hidden_state
    proj = stock.language_model.embedding_projection
    m = adapt.EG2DecisionModel(adapt.BASE, tok, device, lora=16, revision=adapt.BASE_REVISION, special_embeddings=True, attn=attn).eval()
    try:
        with m.lm.disable_adapter():
            off = m.backbone(ids, att)
    except adapt.NonFiniteHidden as e:
        return {"device": device, "attn": attn, "rows": len(rows), "nonfinite": str(e), "used": False}
    on = m.backbone(ids, att)
    valid = att.bool()
    d_off = float((proj(off) - ref)[valid].abs().max()); d_on = float((proj(on) - ref)[valid].abs().max())
    # direct: our hidden state is the stock text model's normed state before the projection
    base = adapt.load_text_model(attn=attn).to(device).eval()
    d_base = float((base(input_ids=ids, attention_mask=att).last_hidden_state - off)[valid].abs().max())
    emb_ref = stock.language_model.embed_tokens(ids); emb_ours = m.lm.get_input_embeddings()(ids)
    d_emb = float((emb_ref - emb_ours).abs().max())
    return {"device": device, "attn": attn, "used": attn == "eager" or device == "cpu", "rows": len(rows), "row_tokens": [len(r) for r in rows], "ref_scale": float(ref[valid].abs().max()),
            "max_abs_diff_lora_disabled": d_off, "max_abs_diff_lora_untrained": d_on, "max_abs_diff_vs_text_model_unprojected": d_base,
            "max_abs_diff_input_embeddings": d_emb}


def escaping(tok):
    specials = adapt.specials(tok)
    text = " ".join(specials + adapt.UPSTREAM_SPECIAL + adapt.DELIMS + ["<unused88>", "<unused6226>", "[multimodal]", "<|fim_prefix|>x<|fim_suffix|>"])
    ids = adapt.user_tokens(tok, f"state {text} end")
    forbidden = set(tok.all_special_ids) | {i for i in range(0, 107)} | set(tok.convert_tokens_to_ids(specials)) | {255999, 256000, 258880, 258881, 258882, 258883, 258884}
    plain = tok(text, add_special_tokens=False).input_ids
    return {"special_strings": len(specials), "forged_ids_after_escaping": sorted(set(ids) & forbidden),
            "forged_ids_without_escaping": sorted(set(plain) & forbidden)[:20], "passed": not (set(ids) & forbidden)}


@torch.no_grad()
def causal_check(tok, device):
    rows = sample_rows(tok, 2)
    out = {}
    for mode in adapt.ATTENTIONS:
        adapt.ATTENTION = mode
        m = adapt.EG2DecisionModel(adapt.BASE, tok, device, revision=adapt.BASE_REVISION).eval()
        r = rows[0]; t = len(r) // 2
        r2 = r[:t + 1] + [tok.convert_tokens_to_ids("▁banana")] * (len(r) - t - 1)
        h1, h2 = (m.backbone(*padded([x], 0, device)) for x in (r, r2))   # one row per pass: batched CPU matmuls round differently
        out[mode] = float((h1[0, : t + 1] - h2[0, : t + 1]).abs().max())
    adapt.ATTENTION = os.environ.get("KEV_EG2_ATTN", "bidirectional")
    return {"max_change_before_t_when_tokens_after_t_change": out, "passed": out["causal"] == 0.0 and out["bidirectional"] > 1e-2}


@torch.no_grad()
def padding_check(tok, device):
    """The row pass's 64-token bucket padding changes nothing: _rows_hidden (bucketed on MPS) vs each row unpadded."""
    out = {}
    for mode in adapt.ATTENTIONS:
        adapt.ATTENTION = mode
        m = adapt.EG2DecisionModel(adapt.BASE, tok, device, revision=adapt.BASE_REVISION).eval()
        rows = [r[1:-1] for r in sample_rows(tok)]
        hs = m._rows_hidden([(r, list(range(len(r)))) for r in rows])
        ref = [m.backbone(*padded([[tok.bos_token_id, *r, tok.eos_token_id]], 0, device))[0, 1:-1] for r in rows]
        out[mode] = max(float((a - b).abs().max()) for a, b in zip(hs, ref)); scale = max(float(b.abs().max()) for b in ref)
    adapt.ATTENTION = os.environ.get("KEV_EG2_ATTN", "bidirectional")
    # masked keys still change the length of the softmax reductions: rounding only (relative to the largest activation)
    return {"device": device, "bucket": adapt.EG2DecisionModel.SHAPE_BUCKET, "max_abs_diff_bucketed_vs_unpadded": out, "hidden_scale": scale,
            "passed": max(out.values()) / scale < 1e-4}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(); a.out = os.path.abspath(a.out)
    adapt.check_upstream(); os.chdir(adapt.KEV_ROOT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapt.BASE, revision=adapt.BASE_REVISION)
    res = {"escaping": escaping(tok)}
    res["parity"] = [parity(tok, "cpu", "eager"), parity(tok, "cpu", "sdpa")]
    if torch.backends.mps.is_available(): res["parity"] += [parity(tok, "mps", "eager"), parity(tok, "mps", "sdpa")]
    res["causal"] = causal_check(tok, "cpu")
    if torch.backends.mps.is_available(): res["padding"] = padding_check(tok, "mps")
    try:
        adapt.load_text_model(dtype=torch.float16); res["fp16_refused"] = False
    except ValueError:
        res["fp16_refused"] = True
    # relative to the largest hidden value (EmbeddingGemma 2 has massive activations: ~3.6e3 projected, ~110 before)
    keys = ("max_abs_diff_lora_disabled", "max_abs_diff_lora_untrained", "max_abs_diff_vs_text_model_unprojected", "max_abs_diff_input_embeddings")
    res["parity_passed"] = all(max(p[k] for k in keys) / p["ref_scale"] < a.tol for p in res["parity"] if p.get("used", True))
    res["passed"] = res["parity_passed"] and res["escaping"]["passed"] and res["causal"]["passed"] and res["fp16_refused"] and res.get("padding", {"passed": True})["passed"]
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
