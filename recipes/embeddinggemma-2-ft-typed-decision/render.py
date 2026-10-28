"""The typed-decision model's text rendering, ported from modeling_gemmadecision.py as a spec for other runtimes.

The rendering is part of the model: every state, question and option becomes a text that the encoder embeds, and
any drift changes the predictions without an error. This is a line-for-line port of the source's `parse_question`,
`render_state`, `render_question`, `render_option` and `GemmaDecisionModel._build_inputs` (8403ff7), written so that
nothing depends on Python's `str()`: `py_str` spells out the formatting the source gets from f-strings, which a
JavaScript port has to reproduce (`True`/`False`/`None`, `1.0`, `1e-07`, `['a', "b's"]`, `{'k': 1}`).
`fixtures.py` writes golden files from it, and `evaluate.py` checks it against the source's own functions on every
evaluated request.

    texts, layout = build(request, workflow="")   # unique texts in order, and the head's index tensors
"""
import json
import math

QUESTION_TYPES = ("choice", "score", "noul")
PROMPT = "task: classification | query: "
MAX_TOKENS = 8192            # truncation length, <bos> and <eos> included
OPTION_PAD = -1e4            # logit of a padded option


# ---- Python's str() / repr() for JSON values -------------------------------------------------------------------
def py_repr_str(s):
    """repr() of a str: single quotes unless the text has ' and no ", backslash escapes, \\x/\\u/\\U for non-printables."""
    quote = '"' if ("'" in s and '"' not in s) else "'"
    out = [quote]
    for ch in s:
        if ch == quote or ch == "\\":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch.isprintable():   # Unicode: not in Cc, Cf, Cs, Co, Cn, Zl, Zp, or Zs other than U+0020
            out.append(ch)
        else:
            o = ord(ch)
            out.append(f"\\x{o:02x}" if o < 0x100 else f"\\u{o:04x}" if o < 0x10000 else f"\\U{o:08x}")
    out.append(quote)
    return "".join(out)


def py_float(x):
    """repr() of a float: the shortest round-trip digits, fixed notation for exponents -4..15, else d.ddde±XX."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "-0.0" if math.copysign(1, x) < 0 else "0.0"
    digits, exp = _shortest(abs(x))          # x = 0.d1d2... * 10**exp
    sign = "-" if x < 0 else ""
    e = exp - 1                               # scientific exponent
    if -4 <= e < 16:
        if e >= len(digits) - 1:
            s = digits + "0" * (e - len(digits) + 1) + ".0"
        elif e >= 0:
            s = digits[: e + 1] + "." + digits[e + 1:]
        else:
            s = "0." + "0" * (-e - 1) + digits
        return sign + s
    mant = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{mant}e{'-' if e < 0 else '+'}{abs(e):02d}"


def _shortest(x):
    # shortest digits that round-trip, like JavaScript's Number.prototype.toString / Python's repr
    for p in range(1, 18):
        s = f"{x:.{p - 1}e}"
        if float(s) == x:
            break
    m, e = s.split("e")
    return m.replace(".", "").rstrip("0") or "0", int(e) + 1


def py_repr(v):
    if isinstance(v, str):
        return py_repr_str(v)
    return py_str(v)


def py_str(v):
    """str() of a JSON value as the source's f-strings produce it."""
    if v is None:
        return "None"
    if v is True:
        return "True"
    if v is False:
        return "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return py_float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "[" + ", ".join(py_repr(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{py_repr(k)}: {py_repr(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"not a JSON value: {type(v).__name__}")


# ---- schema and rendering ---------------------------------------------------------------------------------------
def _json(x):
    return json.loads(x) if isinstance(x, str) else x


def _is_int(k):
    # str(k).lstrip("-").isdigit(): Unicode digits count too (isdigit, not isdecimal)
    return py_str(k).lstrip("-").isdigit()


def parse_question(qid, q):
    """{type, instructions, criteria} -> (kind, instructions, options, descriptions)."""
    kind = q.get("type") or "choice"
    if kind not in QUESTION_TYPES:
        kind = "choice"
    crit = q.get("criteria")
    if kind == "noul":                         # criteria are ignored
        opts, desc = ["false", "true"], ["no", "yes"]
    elif kind == "score":
        if isinstance(crit, list):
            opts, desc = [str(i) for i in range(len(crit))], [py_str(c) for c in crit]
        elif isinstance(crit, dict) and all(_is_int(k) for k in crit):
            keys = sorted(crit, key=int)
            opts, desc = [py_str(k) for k in keys], [py_str(crit[k]) for k in keys]
        elif isinstance(crit, dict):
            opts, desc = [str(i) for i in range(len(crit))], [f"{py_str(k)}: {py_str(v)}" for k, v in crit.items()]
        else:
            opts, desc = [], []
    else:
        if isinstance(crit, dict):
            opts, desc = [py_str(k) for k in crit], [py_str(v) for v in crit.values()]
        elif isinstance(crit, list):
            opts = desc = [py_str(c) for c in crit]
        else:
            opts = desc = []
    if not opts:
        raise ValueError(f"question {qid!r} has no options")
    instr = q.get("instructions")
    return {"qid": qid, "kind": kind, "instructions": py_str(instr) if instr else qid,
            "options": opts, "descriptions": desc}


def render_state(workflow, state):
    if isinstance(state, str) and state.strip()[:1] in "{[":   # "" too: json.loads("") fails, as in the source
        state = json.loads(state)
    if not isinstance(state, dict):
        return f"workflow: {workflow} | state: {py_str(state)}"
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix} · {k}" if prefix else k, v)
        else:
            lines.append(f"{prefix.replace('_', ' ')}: {py_str(value)}")

    walk("", state)
    return f"workflow: {workflow}\n" + "\n".join(lines)


def render_question(spec):
    return f"{spec['kind']} question: {spec['instructions']}"


def render_option(spec, i):
    key, desc = spec["options"][i], spec["descriptions"][i]
    if spec["kind"] == "noul":
        answer = desc
    elif spec["kind"] == "score":
        answer = f"level {key} of {len(spec['options']) - 1}: {desc}"
    else:
        answer = key if desc == key else f"{key}: {desc}"
    return f"question: {spec['instructions']} | answer: {answer}"


def build(request, workflow=""):
    """A System One request ({state, questions}) -> (unique texts, layout for the head graph)."""
    bank = {}

    def tid(t):
        return bank.setdefault(t, len(bank))

    specs, state_index, question_index, options = [], [], [], []
    s = tid(render_state(workflow, request["state"]))
    for qid, q in _json(request["questions"]).items():
        spec = parse_question(qid, q)
        specs.append(spec)
        state_index.append(s)
        question_index.append(tid(render_question(spec)))
        options.append([tid(render_option(spec, i)) for i in range(len(spec["options"]))])
    k = max(len(o) for o in options)
    layout = {
        "specs": specs,
        "state_index": state_index,
        "question_index": question_index,
        "option_index": [o + [0] * (k - len(o)) for o in options],
        "option_mask": [[True] * len(o) + [False] * (k - len(o)) for o in options],
        "question_type": [QUESTION_TYPES.index(sp["kind"]) for sp in specs],
    }
    return list(bank), layout


def answers(specs, probs):
    """Calibrated probabilities (per question, padded) -> the source's TypedAnswer dicts."""
    out = {}
    for spec, p in zip(specs, probs):
        p = [float(x) for x in p[: len(spec["options"])]]
        j = max(range(len(p)), key=lambda i: (p[i], -i))      # first maximum, like torch.argmax
        label = spec["options"][j]
        kind = spec["kind"]
        value = (label == "true") if kind == "noul" else int(label) if kind == "score" else label
        expected = sum(int(o) * pi for o, pi in zip(spec["options"], p)) if kind == "score" else None
        out[spec["qid"]] = {"kind": kind, "label": label, "value": value, "confidence": p[j],
                            "probabilities": dict(zip(spec["options"], p)), "expected_level": expected}
    return out
