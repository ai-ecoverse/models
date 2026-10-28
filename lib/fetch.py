"""Resumable, verified download of a Hub repo snapshot into a plain directory.

Each file is fetched over HTTP with Range requests into <file>.part, resumed after any error, and only renamed into
place once its size and SHA-256 match the Hub's metadata (LFS sha256 for large files, git blob sha1 for small ones).
Existing complete files are re-hashed and kept if they match.

    python fetch.py REPO REVISION OUTDIR [--include GLOB ...] [--exclude GLOB ...] [--repo-type dataset]
"""
import argparse, fnmatch, hashlib, os, sys, time
import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers


def git_sha1(path):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def matches(sib, path):
    if sib.size is not None and os.path.getsize(path) != sib.size:
        return False
    if sib.lfs:
        return sha256(path) == sib.lfs.sha256
    return git_sha1(path) == sib.blob_id


def fetch(repo, rev, sib, out, session, repo_type=None):
    dest = os.path.join(out, sib.rfilename)
    if os.path.exists(dest) and matches(sib, dest):
        print(f"ok (present) {sib.rfilename}", flush=True)
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    url = hf_hub_url(repo, sib.rfilename, revision=rev, repo_type=repo_type)
    headers = build_hf_headers()
    total = sib.size
    t0 = last = time.time()
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have >= total:
            break
        try:
            h = dict(headers, Range=f"bytes={have}-") if have else headers
            with session.get(url, headers=h, stream=True, timeout=60, allow_redirects=True) as r:
                if have and r.status_code != 206:
                    raise IOError(f"no range support ({r.status_code})")
                r.raise_for_status()
                with open(part, "ab") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
                        now = time.time()
                        if now - last > 60:
                            done = f.tell()
                            print(f"  {sib.rfilename}: {done / 1e9:.2f}/{(total or 0) / 1e9:.2f} GB, "
                                  f"{(done - have) / max(now - t0, 1) / 1e6:.2f} MB/s", flush=True)
                            last = now
            if total is None:
                break
        except (requests.RequestException, IOError) as e:
            print(f"  retry {sib.rfilename} at {have}: {e.__class__.__name__}", flush=True)
            time.sleep(10)
            t0 = time.time()
    if not matches(sib, part):
        os.remove(part)
        raise SystemExit(f"checksum mismatch for {sib.rfilename}; removed the partial file, run again")
    os.replace(part, dest)
    print(f"ok {sib.rfilename} ({os.path.getsize(dest) / 1e9:.2f} GB, verified)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("out")
    ap.add_argument("--include", nargs="*", default=["*"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    a = ap.parse_args()
    info = HfApi().repo_info(a.repo, revision=a.revision, repo_type=a.repo_type, files_metadata=True)
    sibs = [s for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, p) for p in a.include)
            and not any(fnmatch.fnmatch(s.rfilename, p) for p in a.exclude)]
    sibs.sort(key=lambda s: s.size or 0)
    with requests.Session() as session:
        for s in sibs:
            fetch(a.repo, info.sha, s, a.out, session, a.repo_type)
    print(f"all {len(sibs)} files verified at {info.sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
