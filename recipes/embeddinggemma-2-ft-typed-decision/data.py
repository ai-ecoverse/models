"""The two evaluation sets, fetched at run time and pinned (neither is stored in this repo).

  typed-decisions  LocalLLaMA/typed-decisions, subset `all`, split `test`: 400 cases, 2,000 decisions with soft gold
                   labels (lib/fetch.py --repo-type dataset, at recipe.yaml's dataset.revision)
  transfer-v4      Kev's held-out transfer-v4 development set (jaredpalmer/kev at recipe.yaml's kev.revision),
                   the 300 records kev.js and ortllama report Kev on: kev.js
                   export/kev_web_export/fixtures.py load_dev(suite="evals/v4/transfer-v4", n=300, seed=1)

    python data.py github --repo ai-ecoverse/kev.js --revision SHA --paths fixtures/kev-0.8b-transfer-v4-dev300.json --out kevjs
    python data.py kev --repo jaredpalmer/kev --revision SHA --path evals/v4/transfer-v4/development.jsonl \
        --sha256 HEX --out kev/development.jsonl
"""
import argparse, hashlib, json, os, random, sys, time


def typed_decisions(parquet):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    return [{"name": r["id"], "workflow": r["workflow"],
             "request": {"state": json.loads(r["state"]), "questions": json.loads(r["questions"])},
             "gold": json.loads(r["gold"])} for r in rows]


def load_dev(path, n=300, seed=1):
    """kev.js's load_dev, verbatim: shuffle with Random(seed), then round-robin over sources until n."""
    rows = [json.loads(line) for line in open(path)]
    random.Random(seed).shuffle(rows)
    by_src, out = {}, []
    for r in rows:
        by_src.setdefault(r["_meta"]["source"], []).append(r)
    while len(out) < n and any(by_src.values()):
        for src in list(by_src):
            if by_src[src] and len(out) < n:
                out.append(by_src[src].pop())
    return [{"name": r["_meta"]["id"], "source": r["_meta"]["source"], "request": {"state": r["state"], "questions": {
        qid: {k: v for k, v in q.items() if k not in ("label", "src", "target")} for qid, q in r["questions"].items()}},
        "labels": {qid: q["label"] for qid, q in r["questions"].items()}} for r in out]


def selection_digest(records):
    return hashlib.sha256("\n".join(r["name"] for r in records).encode()).hexdigest()


def label_index(spec, label):
    """Kev's label (choice key, noul bool, score level) -> index into the typed-decision model's options."""
    if spec["kind"] == "noul":
        return spec["options"].index("true" if label in (True, "true", "True", 1) else "false")
    return spec["options"].index(str(label))


def fetch_github(repo, revision, path, sha256, out):
    import requests
    if os.path.exists(out) and (sha256 is None or hashlib.sha256(open(out, "rb").read()).hexdigest() == sha256):
        print(f"ok (present) {out}", flush=True)
        return
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  retry {url}: {e.__class__.__name__}", flush=True)
            time.sleep(10)
    else:
        raise SystemExit(f"could not fetch {url}")
    got = hashlib.sha256(r.content).hexdigest()
    if sha256 and got != sha256:
        raise SystemExit(f"{url}: sha256 {got}, recipe.yaml pins {sha256}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "wb").write(r.content)
    print(f"ok {out} ({len(r.content)} bytes, verified)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kev")
    for f in ("repo", "revision", "path", "sha256", "out"):
        k.add_argument("--" + f, required=True)
    k.add_argument("--selection-sha256", help="sha256 of the 300 selected record ids, newline-joined")
    g = sub.add_parser("github", help="files of a public GitHub repo at a commit")
    for f in ("repo", "revision", "out"):
        g.add_argument("--" + f, required=True)
    g.add_argument("--paths", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "github":
        for p in a.paths:
            fetch_github(a.repo, a.revision, p, None, os.path.join(a.out, os.path.basename(p)))
        return
    fetch_github(a.repo, a.revision, a.path, a.sha256, a.out)
    recs = load_dev(a.out)
    d = selection_digest(recs)
    print(f"transfer-v4: {len(recs)} records, {sum(len(r['labels']) for r in recs)} questions, selection {d}")
    if a.selection_sha256 and d != a.selection_sha256:
        raise SystemExit(f"selection {d} is not kev.js's ({a.selection_sha256})")


if __name__ == "__main__":
    sys.exit(main())
