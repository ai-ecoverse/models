"""Upload a recipe's out/ folder to a Hugging Face model repo, then verify it.

    python publish.py --repo ai-ecoverse/Foo-ONNX --dir OUT [--path SUBDIR] [--include GLOB ...] \
        [--message MSG] [--private] [--fresh /tmp/verify] [--report verify.json]

upload_folder (huggingface_hub 2.x) is resumable and splits large uploads into several commits; files that are
already on the Hub with the same content are skipped, so re-publishing an unchanged build makes no commit. Every
matching local file is then checked three ways: its size and SHA-256 against the Hub's metadata (LFS sha256, or the
git blob sha1 for small files), and against a fresh download at the new revision into an empty directory (deleted
afterwards).
"""
import argparse, fnmatch, hashlib, json, os, shutil, sys, time
from huggingface_hub import HfApi, hf_hub_download

IGNORE = [".cache/*", "*.part"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_sha1(path):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def local_files(root, include):
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if any(fnmatch.fnmatch(rel, p) for p in IGNORE):
                continue
            if any(fnmatch.fnmatch(rel, p) for p in include):
                out.append(rel)
    return sorted(out)


def upload(api, a):
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
    for attempt in range(6):
        try:
            info = api.upload_folder(repo_id=a.repo, folder_path=a.dir, path_in_repo=a.path or None, repo_type="model",
                                     allow_patterns=a.include, ignore_patterns=IGNORE, commit_message=a.message)
            print("uploaded", info, flush=True)
            return
        except Exception as e:
            print(f"upload attempt {attempt} failed: {e!r}; resuming", flush=True)
            time.sleep(30)
    raise SystemExit("upload failed")


def verify(api, a):
    info = api.model_info(a.repo, files_metadata=True)
    hub = {s.rfilename: s for s in info.siblings}
    rows, ok = [], True
    shutil.rmtree(a.fresh, ignore_errors=True)
    os.makedirs(a.fresh)
    for rel in local_files(a.dir, a.include):
        p = os.path.join(a.dir, rel)
        name = f"{a.path.strip('/')}/{rel}" if a.path else rel
        s = hub.get(name)
        local = sha256(p)
        row = {"file": name, "size": os.path.getsize(p), "sha256": local}
        if s is None:
            row["error"] = "missing on the Hub"
        else:
            row["hub_size_ok"] = s.size == row["size"]
            row["hub_hash_ok"] = (s.lfs.sha256 == local) if s.lfs else (s.blob_id == git_sha1(p))
            d = hf_hub_download(a.repo, name, revision=info.sha, local_dir=a.fresh)
            row["redownload_ok"] = sha256(d) == local
            os.remove(d)
        good = "error" not in row and row["hub_size_ok"] and row["hub_hash_ok"] and row["redownload_ok"]
        ok &= good
        rows.append(row)
        print(("ok  " if good else "BAD ") + name, row["size"], local[:16], flush=True)
    shutil.rmtree(a.fresh, ignore_errors=True)
    if a.report:
        json.dump({"repo": a.repo, "revision": info.sha, "ok": ok, "files": rows}, open(a.report, "w"), indent=1)
    print("verified" if ok else "VERIFY FAILED", f"https://huggingface.co/{a.repo}/tree/{info.sha}", flush=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"revision={info.sha}\n")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--path", default="", help="subfolder of the Hub repo to publish into")
    ap.add_argument("--include", nargs="+", default=["*"])
    ap.add_argument("--message", default="Upload ONNX export")
    ap.add_argument("--private", action="store_true", help="create the repo private if it does not exist yet")
    ap.add_argument("--fresh", default="/tmp/models-verify")
    ap.add_argument("--report")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    if not a.verify_only:
        upload(api, a)
    return verify(api, a)


if __name__ == "__main__":
    sys.exit(main())
