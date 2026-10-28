"""EmbeddingGemma 2 as the backbone of upstream Kev's decision recipe: the whole adaptation, applied to a pinned,
unmodified checkout of jaredpalmer/kev at import time (`import kev_eg2.adapt` before using kev.*).

What changes against upstream (kev.model / kev.checkpoint), and why:

1. Backbone. Upstream loads `AutoModelForCausalLM(...).model`. EmbeddingGemma 2 is `EmbeddingGemma2Model`; only its
   text model is built (vision_config / audio_config off, so those towers are never instantiated) and its
   `embedding_projection` (512 -> 768, applied per token after the final norm) is replaced by the identity: the
   pointer head reads the normed 512-d last hidden state, before pooling and projection.
2. Attention. EmbeddingGemma 2 is a bidirectional encoder (`is_causal = False`; bidirectional masks on every layer,
   +-512 tokens on the sliding layers). transformers passes a 4D mask through unchanged to *both* layer types
   (masking_utils._preprocess_mask_arguments), which would silently drop the sliding window, and there is no
   key-value cache, so upstream's packed block-causal form and its state-prefix reuse do not apply. Every question runs
   as its own row (upstream's row form, kev.model.rows_of: state + that question's branch), wrapped in <bos> ... <eos>
   as the encoder was trained. ATTENTION = "bidirectional" (the model's own masks from a 2D padding mask) or "causal"
   (a causal mask per layer type, the sliding window kept): the one extra arm the protocol allows.
3. Delimiters. Upstream's SPECIAL are Qwen strings. They map to <unused0..4> (ids 6-10), which no Gemma template
   emits and BPE cannot produce from text; their embedding rows are trained (peft trainable_token_indices, upstream's
   --special_embeddings 1). Caller text is escaped with upstream's rule (`<|name|>` -> `<¦name¦>`) and then every one
   of Gemma's added (special) tokens left in it is broken with a word joiner, so user text cannot forge a control token.
4. LoRA targets. Upstream's "all" list (q/k/v/o, gate/up/down) matches Gemma 4's names; the per-layer-input (PLE)
   projections stay frozen.
5. Precision. fp16 is refused (EmbeddingGemma 2 overflows to NaN in fp16 without an error). Weights are fp32; every
   backbone pass is checked for NaN/inf and fails the run.
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KEV_ROOT = Path(os.environ.get("KEV_ROOT", HERE / ".upstream" / "kev")).resolve()
KEV_COMMIT = "2074cc44256200bb37540103fa4572901faa10f3"
if str(KEV_ROOT) not in sys.path: sys.path.insert(0, str(KEV_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import kev.model as km
import kev.checkpoint as kc

BASE = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
UPSTREAM_SPECIAL = list(km.SPECIAL)
DELIMS = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]   # state, question, <opt>, </opt>, <decide>
ATTENTIONS = ("bidirectional", "causal")
ATTENTION = os.environ.get("KEV_EG2_ATTN", "bidirectional")
if ATTENTION not in ATTENTIONS: raise ValueError(f"KEV_EG2_ATTN must be one of {ATTENTIONS}")
WORD_JOINER = "\u2060"

km.SPECIAL[:] = DELIMS   # in place: kev.model.encode reads the module global
_upstream_user_tokens = km.user_tokens
_specials = {}


def specials(tok):
    """Gemma's added (special) token strings, longest first (the tokenizer splits them out of plain text)."""
    key = id(tok)
    if key not in _specials:
        _specials[key] = sorted({t.content for t in tok.added_tokens_decoder.values() if len(t.content) > 1}, key=len, reverse=True)
    return _specials[key]


def user_tokens(tok, text):
    """Upstream's escaping (`<|name|>` -> `<¦name¦>`), then every remaining special-token string broken after its first
    character with a word joiner: caller text never yields a control or delimiter token."""
    text = km._SPECIAL_RE.sub(r"<¦\1¦>", text)
    for s in specials(tok):
        if s in text: text = text.replace(s, s[0] + WORD_JOINER + s[1:])
    return tok(text, add_special_tokens=False).input_ids


km.user_tokens = user_tokens


def load_text_model(name=BASE, revision=BASE_REVISION, dtype=torch.float32, attn="eager"):
    """EmbeddingGemma 2's text encoder alone, `embedding_projection` removed (identity)."""
    if dtype == torch.float16: raise ValueError("EmbeddingGemma 2 overflows to NaN in fp16: use fp32 (or bf16)")
    from transformers import AutoConfig
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2Model
    cfg = AutoConfig.from_pretrained(name, revision=revision)
    cfg.vision_config = None; cfg.audio_config = None
    full = EmbeddingGemma2Model.from_pretrained(name, revision=revision, config=cfg, dtype=dtype, attn_implementation=attn)
    lm = full.language_model
    lm.embedding_projection = nn.Identity()
    return lm


def row_masks(att, sliding_window, dtype):
    """Causal arm: additive [B, 1, L, L] masks per layer type (causal, padded keys masked, the sliding layers' window
    kept as transformers' causal sliding mask has it: q - k < window). Padded queries keep their causal keys."""
    L = att.shape[1]
    i = torch.arange(L, device=att.device)
    causal = i[None, :] <= i[:, None]
    valid = att.bool()[:, None, :]
    full = causal[None] & valid
    sliding = full & ((i[:, None] - i[None, :]) < sliding_window)[None]
    neg = torch.finfo(dtype).min
    as_bias = lambda allow: torch.zeros(allow.shape, dtype=dtype, device=att.device).masked_fill(~allow, neg)[:, None]
    return {"full_attention": as_bias(full), "sliding_attention": as_bias(sliding)}


class NonFiniteHidden(FloatingPointError):
    pass


class EG2DecisionModel(km.DecisionModel):
    """kev.model.DecisionModel on EmbeddingGemma 2's text encoder (rows only; see the module docstring). Same
    signature as upstream's constructor, so kev.checkpoint and kev.train build it unchanged."""

    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False,
                 lora_targets="all", dtype=torch.float32, weights=None, direct_load=False):
        nn.Module.__init__(self)
        if option_isolation: raise ValueError("option_isolation needs the packed mask; EmbeddingGemma 2 runs rows")
        if weights or direct_load: raise ValueError("full-weight checkpoints are not part of this experiment")
        if name != BASE: raise ValueError(f"this adaptation is for {BASE}, not {name}")
        # upstream's rule (SDPA on CUDA, eager elsewhere); on MPS, SDPA returns NaN for padded rows (runs/checks.json)
        attn = attn or os.environ.get("KEV_EG2_IMPL") or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.lm = load_text_model(name, revision or BASE_REVISION, dtype=dtype, attn=attn)
        self.pad_id = km.pad_id(tok)
        self.bos, self.eos = tok.bos_token_id, tok.eos_token_id
        self.delims = [tok.convert_tokens_to_ids(t) for t in DELIMS]
        if self.delims != [6, 7, 8, 9, 10]: raise ValueError(f"unexpected delimiter ids {self.delims}")
        self.hybrid, self.option_isolation, self.attention = False, False, ATTENTION
        self.sliding_window = self.lm.config.sliding_window
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": list(self.delims)}} if special_embeddings else {}
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = km.PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    @property
    def prefix_min_tokens(self):
        return 1 << 62   # never cache a state prefix: an encoder has none

    def rows_form(self, encs):
        return True

    def backbone(self, ids, att):
        """Last hidden states [B, L, d] (fp32) of right-padded rows under this model's attention."""
        mask = att if self.attention == "bidirectional" else row_masks(att, self.sliding_window, next(self.lm.parameters()).dtype)
        pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        h = self.lm(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state.float()
        if not torch.isfinite(h).all(): raise NonFiniteHidden("non-finite hidden states from the EmbeddingGemma 2 backbone")
        return h

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """Upstream's row pass, each row wrapped in <bos> ... <eos>; the returned states are the row's own tokens (the
        wrapper stripped), so every upstream readout offset holds. One row per backbone pass, right-padded to a multiple
        of SHAPE_BUCKET on MPS (upstream's eval-time bucketing, here in training too): MPS keeps per-shape kernels and
        buffers, and with free (rows x length) shapes its driver allocation grew past 90 GB in 200 passes (runs/prof);
        one row at a time in 64-token buckets it levels off near 9 GB. Rows are independent and pads are masked keys
        placed after every real token, so the hidden states do not depend on either choice."""
        if cache is not None: raise ValueError("EmbeddingGemma 2 has no key-value cache")
        bucket = self.SHAPE_BUCKET if str(self.device) == "mps" else 1
        out = []
        for rid, rpos in rows:
            if list(rpos) != list(range(len(rid))): raise ValueError("row positions are not contiguous")
            L = -(-(len(rid) + 2) // bucket) * bucket
            ids = torch.full((1, L), self.pad_id, dtype=torch.long, device=self.device)
            att = torch.zeros((1, L), dtype=torch.long, device=self.device)
            ids[0, : len(rid) + 2] = torch.tensor([self.bos, *rid, self.eos], device=self.device); att[0, : len(rid) + 2] = 1
            out.append(self.backbone(ids, att)[0, 1 : 1 + len(rid)])
        return out

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    def prefix(self, enc):
        raise NotImplementedError("no state-prefix reuse on an encoder")

    def probs_and_prefix(self, enc):
        return self.probs(enc), None

    def probs_with_prefix(self, enc, prefix):
        raise NotImplementedError("no state-prefix reuse on an encoder")

    def probs_batch(self, encs, prefixes, keep):
        return [self.probs(e) for e in encs], [None] * len(encs)


# kev.checkpoint (and so kev.predictors, kev.benchmark, kev.experiment) and kev.train build the model by these names
km.DecisionModel = EG2DecisionModel
kc.DecisionModel = EG2DecisionModel


def check_upstream():
    """The checkout is the pinned commit, unmodified (our changes live only here)."""
    import subprocess
    head = subprocess.run(["git", "-C", str(KEV_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(KEV_ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    if head != KEV_COMMIT: raise SystemExit(f"{KEV_ROOT} is at {head}, expected {KEV_COMMIT}")
    if dirty: raise SystemExit(f"{KEV_ROOT} has local modifications:\n{dirty}")
    return head
