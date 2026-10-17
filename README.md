# models

Conversion recipes for the ONNX models ai-ecoverse publishes on
[Hugging Face](https://huggingface.co/ai-ecoverse), for
[slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama), transformers.js
and plain onnxruntime. Each recipe pins its source checkpoint to a Hub commit,
converts it in CI, checks the result against PyTorch, and uploads it along with
a copy of the recipe that made it.

## Layout

```text
recipes/<name>/
  recipe.yaml       # source repo + revision, target repo, runner, timeout, publish
  requirements.txt  # pinned Python packages (uv venv per recipe)
  convert.sh        # recipe body: runs in $WORK, writes the Hub repo's files to $OUT
  *.py, card/ …     # whatever the conversion needs, plus the model card template
lib/                # shared Python for recipes (fetch.py: verified, resumable Hub download)
scripts/
  run.sh            # run one recipe: venv, convert.sh, export/ + provenance.json
  publish.py        # upload_folder to the Hub, then verify every file (SHA-256, fresh download)
  read-recipe.mjs   # recipe.yaml fields (--field, --json, --env)
  list-touched-recipes.mjs
```

`convert.sh` gets `RECIPE_DIR`, `WORK` (scratch, kept between local runs),
`OUT` (becomes the Hub repo, or `target.path` inside it), `RESULTS` (parity
numbers and logs, uploaded as a CI artifact, not to the Hub), `LIB`, and every
recipe field as `RECIPE_*` (`RECIPE_SOURCE_REPO`, `RECIPE_SOURCE_REVISION`,
lists space-separated). It should fail when the export drifts from the source
model: each recipe has a parity gate.

`run.sh` then copies the recipe and `lib/` into `$OUT/export/`, with the
resolved `environment.txt` and a `provenance.json` holding this repo's commit
and the CI run, so every published model carries the exact recipe that made it.

## Recipes

| Recipe | Hub repo | Runner | Notes |
| --- | --- | --- | --- |
| [`ministral-3-3b-reasoning-2512`](recipes/ministral-3-3b-reasoning-2512) | [ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX) | `gce-xl` | Reasoning weights in Mistral's Instruct ONNX graphs; 5 variants, ~36 GB; parity gate against PyTorch fp32 |
| [`tiny-random-llama`](recipes/tiny-random-llama) | — (`publish: false`) | `ubuntu-latest` | Pipeline smoke test; template for plain `torch.onnx.export` recipes |

## Running locally

Needs Node, [uv](https://docs.astral.sh/uv/) and the disk and RAM the recipe
asks for in `recipe.yaml`.

```bash
bash scripts/run.sh tiny-random-llama          # → .models-out/tiny-random-llama/out
FORCE=1 bash scripts/run.sh tiny-random-llama  # clear out/ and results/ first
node scripts/read-recipe.mjs ministral-3-3b-reasoning-2512 --json
```

Publish a local build by hand (needs `HF_TOKEN` or `hf auth login`):

```bash
uv run --no-project --with huggingface_hub==2.2.0 --with hf-xet==1.7.0 \
  python scripts/publish.py --repo ai-ecoverse/Foo-ONNX --dir .models-out/foo/out
```

## CI

| Workflow | When | Does |
| --- | --- | --- |
| [`check`](.github/workflows/check.yml) | PRs touching `recipes/`, `lib/`, `scripts/`; or dispatch | Converts touched recipes on `ubuntu-latest` (lib/scripts changes run `tiny-random-llama`); never publishes, no HF token. Recipes on other runners are skipped on PRs: dispatch `check` with `recipe=<name>` on the branch to convert one (`runner=` overrides the recipe's runner). |
| [`convert`](.github/workflows/convert.yml) | dispatch `recipe=<name>` (`publish` defaults to true) | Converts on the recipe's `runner`, uploads results as an artifact, publishes to `target.repo` and verifies the upload. Runs in the `huggingface` environment, which only deploys from `main`. |
| [`convert-merge`](.github/workflows/convert-merge.yml) | push to `main` touching `recipes/<name>/` | Dispatches `convert` for each touched recipe. |

A recipe change that should not republish (runner, timeout, comments) can go
in with `[skip ci]` in the commit message, which skips `convert-merge`.

Re-publishing an unchanged build is safe: `upload_folder` skips files that are
already on the Hub, so it makes no commit.

### Runners

`runner:` in `recipe.yaml` is the `runs-on` label.

- `ubuntu-latest`: free for this public repo; 4 vCPU, 16 GB RAM, ~14 GB
  disk (more after the workflow's cleanup step). Enough for models up to
  roughly 1B parameters.
- `gce-xl`: one Compute Engine VM per job in GCP project `helix-research`
  (n2-highmem-8: 8 vCPU, 64 GB RAM, 300 GB disk, on demand, about $0.55 an
  hour), started by [gh-runner-broker](https://github.com/ai-ecoverse/gh-runner-broker)
  when a job asks for the label and deleted when the job ends. Runner group
  `gce-xl`, open to this repo only; at most 2 VMs at a time. Used for
  3B-class models.

The Cloud Run runners (`cloud-run-gpu`, `cloud-run-bench`, …) stop at 8 vCPU
and 32 GiB with the filesystem in RAM, so they can't hold the ~80 GB a
3B-model conversion writes.

## Secrets

| Where | Name | What |
| --- | --- | --- |
| environment `huggingface` | `HF_TOKEN` | Hugging Face token with write access to the `ai-ecoverse` org |

## Adding a recipe

1. Copy `recipes/tiny-random-llama` (plain PyTorch export) or a recipe closer
   to the model at hand.
2. Pin `source.revision` to a Hub commit, not a branch.
3. Make `convert.sh` fail when parity with the source model is off.
4. Run it locally, or open a PR: `check` converts it if it fits on
   `ubuntu-latest`.
5. Merge. `convert-merge` dispatches `convert`, which publishes.

## License

Apache-2.0 (recipes and tooling). Models keep their own licenses, stated on
their model cards.
