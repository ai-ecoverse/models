"""Check the response template in the assembled tokenizer_config.json against response_cases.json.

    python response_check.py --repo OUT --cases response_cases.json --out results/response-template.json

Loads the export's tokenizer with transformers and, for every case, requires tokenizer.parse_response (prefix: the
case's prompt from the export's chat template) and a streamed ResponseParser fed one token at a time to return the
case's expected message. When the reply finished (and the case doesn't say "round_trip": false), the parsed message
appended to the conversation must render back to the same reply. scripts/ortllama-gate.mjs holds slicc-ortllama's own parser to the same expected messages.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def streamed(tok, raw, prefix, tools):
    parser = tok.get_response_parser(prefix=prefix, tools=tools)
    ids = tok.encode(raw, add_special_tokens=False)
    text = ""
    for i in range(len(ids)):
        now = tok.decode(ids[: i + 1], skip_special_tokens=False)
        parser.feed(now[len(text):])
        text = now
    message, _ = parser.finalize()
    return message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cases", default=os.path.join(HERE, "response_cases.json"))
    ap.add_argument("--template", default=os.path.join(HERE, "response_template.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.repo)
    problems = []
    if tok.response_template != json.load(open(a.template)):
        problems.append("tokenizer_config.json's response_template is not response_template.json")
    rows = []
    for c in json.load(open(a.cases))["cases"]:
        tools, raw, want = c.get("tools"), c["raw"], c["expected"]
        prefix = tok.apply_chat_template(c["messages"], tools=tools, add_generation_prompt=True, tokenize=False)
        row = {"case": c["name"]}
        try:
            row["one_shot"] = tok.parse_response(raw, prefix=prefix, tools=tools) == want
            row["streamed"] = streamed(tok, raw, prefix, tools) == want
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        if raw.endswith(tok.eos_token) and c.get("round_trip", True) and "error" not in row:
            again = tok.apply_chat_template(c["messages"] + [want], tools=tools, tokenize=False)
            row["round_trip"] = again == prefix + raw
        bad = [k for k in ("one_shot", "streamed", "round_trip") if row.get(k) is False] + (["error"] if "error" in row else [])
        if bad:
            problems.append(f"{c['name']}: {', '.join(bad)}")
        rows.append(row)
        print(f"{'BAD' if bad else 'ok '} {c['name']}" + (f" ({', '.join(bad)})" if bad else ""), flush=True)
    json.dump({"cases": rows, "problems": problems}, open(a.out, "w"), indent=1)
    if problems:
        raise SystemExit("response template check failed:\n  " + "\n  ".join(problems))
    print(f"response template: {len(rows)} case(s) parse as expected, one-shot and streamed")


if __name__ == "__main__":
    sys.exit(main())
