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
  recipe.yaml       # source repo + revision, target repo, runner, timeout, publish, ortllama
  requirements.txt  # pinned Python packages (uv venv per recipe)
  convert.sh        # recipe body: runs in $WORK, writes the Hub repo's files to $OUT
  *.py, card/ …     # whatever the conversion needs, plus the model card template
lib/                # shared code for recipes (fetch.py: verified, resumable Hub download, models or datasets)
  ministral3/       # Ministral 3 8B/14B export: genai-builder decoder, re-weighted 3B vision/embed graphs
scripts/
  run.sh            # run one recipe: venv, convert.sh, export/ + provenance.json, ortllama gate
  ortllama-gate.mjs # fail unless slicc-ortllama lists the export as recipe.yaml declares
  ortllama/         # the pinned slicc-ortllama (package.json + lock) the gate runs
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

### slicc-ortllama gate

Last, `run.sh` runs [`scripts/ortllama-gate.mjs`](scripts/ortllama-gate.mjs) on
the finished `$OUT`, the folder that gets published.
[slicc-ortllama](https://github.com/ai-ecoverse/slicc-ortllama) finds models by
file layout alone, so an export with a file name it doesn't know would publish
fine and never show up in `ortllama ls`. The gate installs the slicc-ortllama
pinned in [`scripts/ortllama/`](scripts/ortllama/package.json) (Renovate bumps
it), points `ORTLLAMA_MODEL_PATH` at `$OUT` with an empty Hugging Face cache and
ortllama home, and fails unless:

- `ortllama ls` lists the folder (as `local/<dir>`) with the declared `engine`;
- its variants are exactly the declared `variants`, no more and no fewer;
- every variant resolves with all its files (`embed_tokens`, tokenizer, chat
  template, the external data `config.json` declares), which listing alone
  doesn't check;
- every external data file an `.onnx` graph in the export points to is there
  (ortllama only looks for data files it is told about or that exist);
- with `default:`, the default variant is that one;
- with `check: true`, `ortllama check` loads the default variant and answers one
  request (installs `onnxruntime-node` too; a 3B q4f16 needs ~4 GB RAM);
- with `recipes/<name>/response_cases.json` (recorded replies and the messages
  they parse to), ortllama's own reply parser returns each of those messages.
  ortllama doesn't read a tokenizer's `response_template`; it parses reasoning
  and tool calls itself, so this keeps the two in agreement. The recipe holds
  transformers' `parse_response` to the same file, and an export whose
  `tokenizer_config.json` has a `response_template` fails without it.

The declaration in `recipe.yaml`:

```yaml
# engine: decoder, whisper, parakeet, kokoro, phonon, silero or systemone (kev.js)
# default: optional; variants: ortllama's names, so the _quantized suffix is q8
# check: optional, off by default; only on runners with the RAM
ortllama:
  engine: decoder
  default: q4f16
  variants:
    - q4f16
    - q4
    - fp16
    - q8
    - fp32
  check: true
```

`read-recipe.mjs` reads a small YAML subset: comments go on their own lines, and
lists are `- item` blocks.

A recipe with `publish: true` and no `ortllama:` block fails the gate: every
published export has to say what ortllama should make of it, or opt out with
`ortllama: none`. Without the block and with `publish: false` the gate only
records what ortllama sees. The result, with ortllama's full listing, goes to
`$RESULTS/ortllama.json` next to the parity numbers.

To try the gate on a folder by hand (Hub snapshots are symlinks into `blobs/`,
which ortllama doesn't follow for local folders, so copy one with `cp -RL`
first):

```bash
mkdir -p .models-out/.ortllama && cp scripts/ortllama/package*.json .models-out/.ortllama/
(cd .models-out/.ortllama && npm ci --ignore-scripts --include=optional)
node scripts/ortllama-gate.mjs <recipe> --out <folder> --results /tmp/gate --ortllama .models-out/.ortllama
```

## Recipes

| Recipe | Hub repo | Runner | Notes |
| --- | --- | --- | --- |
| [`embeddinggemma-2-ft-typed-decision`](recipes/embeddinggemma-2-ft-typed-decision) | — (`publish: false`) | `ubuntu-latest` | mrm8488's EmbeddingGemma 2 System One decision model: fine-tuned encoder in onnx-community's text graph (fp32, q8) + decision head graph; self-tests rebuild onnx-community's bytes; parity gate against `model.decide()` on typed-decisions and Kev's transfer-v4 300; golden rendering fixtures; Kev comparison; ortllama gate records only (no System One backend for it yet) |
| [`ministral-3-3b-reasoning-2512`](recipes/ministral-3-3b-reasoning-2512) | [ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-3B-Reasoning-2512-ONNX) | `gce-xl` | Reasoning weights in Mistral's Instruct ONNX graphs; 5 variants, ~36 GB; parity gate against PyTorch fp32; response template checked on recorded replies in transformers and ortllama; ortllama gate with `check` |
| [`ministral-3-8b-instruct-2512`](recipes/ministral-3-8b-instruct-2512) | [ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-8B-Instruct-2512-ONNX) | `gce-xl` | genai-builder decoder (q4f16, q4) + fp16 vision/embeddings ([`lib/ministral3`](lib/ministral3)); from the BF16 checkpoint; ortllama gate with `check` |
| [`ministral-3-8b-reasoning-2512`](recipes/ministral-3-8b-reasoning-2512) | [ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-8B-Reasoning-2512-ONNX) | `gce-xl` | same |
| [`ministral-3-14b-instruct-2512`](recipes/ministral-3-14b-instruct-2512) | [ai-ecoverse/Ministral-3-14B-Instruct-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-14B-Instruct-2512-ONNX) | `gce-xl` | same; ~60 GB RAM for the fp32 parity reference |
| [`ministral-3-14b-reasoning-2512`](recipes/ministral-3-14b-reasoning-2512) | [ai-ecoverse/Ministral-3-14B-Reasoning-2512-ONNX](https://huggingface.co/ai-ecoverse/Ministral-3-14B-Reasoning-2512-ONNX) | `gce-xl` | same |
| [`phonon-31m-en`](recipes/phonon-31m-en) | [ai-ecoverse/phonon-31m-en-ONNX](https://huggingface.co/ai-ecoverse/phonon-31m-en-ONNX) | `ubuntu-latest` | Gradium's Phonon 31m English TTS, reimplemented in PyTorch and exported (flow LM with KV cache, streaming Mimi decoder, 37 voices); parity gate against ptts (Rust, built in CI) and slicc-ortllama's tokenizer; ortllama gate with `check` |
| [`tiny-random-llama`](recipes/tiny-random-llama) | — (`publish: false`) | `ubuntu-latest` | Pipeline smoke test; template for plain `torch.onnx.export` recipes; runs the ortllama gate (no `check`) |
| [`underdog-bark-3.0`](recipes/underdog-bark-3.0) | [ai-ecoverse/Underdog-Bark-3.0-ONNX](https://huggingface.co/ai-ecoverse/Underdog-Bark-3.0-ONNX) | `ubuntu-latest` | Parakeet TDT speech recognition: `model.bark` unpacked with the source's loader, encoder + one-step decoder_joint graphs, fp32/fp16/q8/q4f16; parity gate against transformers' `generate()` on `say` clips; engine fixtures in `fixture/`; ortllama `parakeet` gate with `check` |

## Running locally

Needs Node 24+, [uv](https://docs.astral.sh/uv/) and the disk and RAM the recipe
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
| [`check`](.github/workflows/check.yml) | PRs touching `recipes/`, `lib/`, `scripts/`; or dispatch | Converts touched recipes on `ubuntu-latest` (lib/scripts changes run `tiny-random-llama`), parity and ortllama gates included; never publishes, no HF token. Recipes on other runners are skipped on PRs: dispatch `check` with `recipe=<name>` on the branch to convert one (`runner=` overrides the recipe's runner). |
| [`convert`](.github/workflows/convert.yml) | dispatch `recipe=<name>` (`publish` defaults to true) | Converts on the recipe's `runner` (a failed parity or ortllama gate stops it before publishing), uploads results as an artifact, publishes to `target.repo` and verifies the upload. Runs in the `huggingface` environment, which only deploys from `main`. |
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
  (n2-highmem-16: 16 vCPU, 128 GB RAM, 400 GB disk, on demand, about $1.05 an
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
4. Declare `ortllama:` (engine and variants) if the model is meant for
   slicc-ortllama, or `ortllama: none` if it isn't.
5. Run it locally, or open a PR: `check` converts it if it fits on
   `ubuntu-latest`.
6. Merge. `convert-merge` dispatches `convert`, which publishes.

## License

Apache-2.0 (recipes and tooling). Models keep their own licenses, stated on
their model cards.
