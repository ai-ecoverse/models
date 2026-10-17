#!/usr/bin/env node
/**
 * slicc-ortllama acceptance gate: point a pinned ortllama (scripts/ortllama/) at the
 * export and fail unless `ortllama ls` lists it with the engine and exactly the
 * variants recipe.yaml declares under `ortllama:`, and every one of them resolves
 * with all its files. With `ortllama.check: true` it also runs `ortllama check` on
 * the default variant (loads it and makes one request; needs onnxruntime-node).
 *
 *   node scripts/ortllama-gate.mjs <recipe> --out DIR --results DIR --ortllama DIR
 *
 * --ortllama is the folder `npm ci` installed scripts/ortllama/ into. Writes
 * RESULTS/ortllama.json. Recipes without `ortllama:` fail if they publish; with
 * `publish: false` they only record what ortllama sees. `ortllama: none` opts a
 * published model out on purpose.
 */
import { spawnSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, delimiter, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    out: { type: 'string' },
    results: { type: 'string' },
    ortllama: { type: 'string' },
  },
});
const [name] = positionals;
if (!name || !values.out || !values.results || !values.ortllama) {
  console.error('usage: ortllama-gate.mjs <recipe> --out DIR --results DIR --ortllama DIR');
  process.exit(2);
}
const out = resolve(values.out);
if (out.includes(delimiter)) {
  console.error(`ortllama gate: ${out} contains '${delimiter}', which ORTLLAMA_MODEL_PATH splits on`);
  process.exit(2);
}

const read = spawnSync(process.execPath, [join(root, 'scripts', 'read-recipe.mjs'), name, '--json'], {
  encoding: 'utf8',
});
if (read.status !== 0) {
  process.stderr.write(read.stderr);
  process.exit(2);
}
const recipe = JSON.parse(read.stdout);
const spec = recipe.ortllama;
const publishes = recipe.publish === 'true';

const pkg = join(resolve(values.ortllama), 'node_modules', '@ai-ecoverse', 'slicc-ortllama');
const bin = join(pkg, 'src', 'bin.js');
const { VERSION, openCatalog } = await import(pathToFileURL(join(pkg, 'src', 'index.js')).href);

// Only the export: an empty Hugging Face cache and ortllama home, so neither the
// machine's downloads nor its aliases and hints change what the gate sees.
const scratch = mkdtempSync(join(tmpdir(), 'ortllama-gate-'));
const env = {
  ...process.env,
  ORTLLAMA_MODEL_PATH: out,
  ORTLLAMA_HOME: join(scratch, 'home'),
  HF_HOME: join(scratch, 'hf'),
  HF_HUB_CACHE: join(scratch, 'hf', 'hub'),
};
const ortllama = (...args) =>
  spawnSync(process.execPath, [bin, ...args], { env, encoding: 'utf8', maxBuffer: 64 << 20 });

const id = `local/${basename(out)}`;
const problems = [];
const report = { recipe: name, ortllama: VERSION, id, expected: spec ?? null };
const say = (line) => console.log(`ortllama gate: ${line}`);

function finish() {
  rmSync(scratch, { recursive: true, force: true });
  report.problems = problems;
  report.ok = problems.length === 0;
  mkdirSync(values.results, { recursive: true });
  writeFileSync(join(values.results, 'ortllama.json'), `${JSON.stringify(report, null, 1)}\n`);
  for (const p of problems) say(`FAIL ${p}`);
  say(report.ok ? 'passed' : `failed (${problems.length} problem(s), see ortllama.json)`);
  process.exit(report.ok ? 0 : 1);
}

const ls = ortllama('ls', '--json');
if (ls.status !== 0) {
  problems.push(`ortllama ls exited ${ls.status}: ${ls.stderr.trim()}`);
  finish();
}
const local = JSON.parse(ls.stdout).models.filter((m) => m.id.startsWith('local/'));
const entry = local.find((m) => m.id === id) ?? null;
report.listed = local;
say(
  entry
    ? `ortllama ${VERSION} lists ${id}: engine ${entry.engine}, variants ${entry.variants.join(', ')}, default ${entry.defaultVariant}`
    : `ortllama ${VERSION} does not list ${id}${local.length ? ` (it lists ${local.map((m) => m.id).join(', ')})` : ''}`
);

if (spec === undefined) {
  if (publishes) problems.push(`recipe.yaml has no ortllama: block; a published recipe declares engine and variants (or ortllama: none)`);
  else say('no ortllama: block and publish: false, so this is a record only');
  report.recordOnly = true;
  finish();
}
if (spec === 'none') {
  say('ortllama: none, not an ortllama model');
  finish();
}
if (typeof spec !== 'object' || typeof spec.engine !== 'string' || !Array.isArray(spec.variants) || spec.variants.length === 0) {
  problems.push('ortllama: needs engine: <name> and a non-empty variants: list');
  finish();
}

if (!entry) {
  problems.push(`${id} is not listed, so ortllama would not offer this export at all`);
  finish();
}
if (entry.engine !== spec.engine) problems.push(`engine is ${entry.engine}, recipe.yaml expects ${spec.engine}`);
const missing = spec.variants.filter((v) => !entry.variants.includes(v));
const extra = entry.variants.filter((v) => !spec.variants.includes(v));
if (missing.length) problems.push(`variant(s) ${missing.join(', ')} not listed; ortllama lists ${entry.variants.join(', ')}`);
if (extra.length) problems.push(`variant(s) ${extra.join(', ')} listed but not declared in recipe.yaml`);
if (spec.default && entry.defaultVariant !== spec.default)
  problems.push(`default variant is ${entry.defaultVariant}, recipe.yaml expects ${spec.default}`);

// Listing only needs the graph file; resolving checks every file the variant loads
// (external data, embed_tokens, tokenizer, chat template).
const catalog = await openCatalog({ env });
report.variants = {};
for (const v of spec.variants.filter((x) => entry.variants.includes(x))) {
  try {
    await catalog.resolve(`${id}:${v}`);
    report.variants[v] = { complete: true };
    say(`ok  ${v}`);
  } catch (error) {
    report.variants[v] = { complete: false, error: error.message };
    problems.push(error.message);
  }
}

// ortllama only expects a <graph>_data file when config.json declares it or it
// exists, so a graph whose weights point at a missing file would still resolve.
// Read the locations from the graphs themselves: TensorProto.external_data holds
// StringStringEntryProto { key: "location", value } (fields 1 and 2).
const LOCATION = Buffer.from([0x0a, 0x08, ...Buffer.from('location'), 0x12]);

async function externalData(path) {
  const found = new Set();
  let tail = Buffer.alloc(0);
  for await (const chunk of createReadStream(path, { highWaterMark: 16 << 20 })) {
    const buf = Buffer.concat([tail, chunk]);
    let at = buf.indexOf(LOCATION);
    while (at !== -1) {
      let i = at + LOCATION.length;
      let len = 0;
      for (let shift = 0; i < buf.length && shift < 28; shift += 7) {
        const b = buf[i++];
        len |= (b & 0x7f) << shift;
        if (!(b & 0x80)) break;
      }
      if (i + len > buf.length) break; // value runs past the chunk: picked up from the tail
      found.add(buf.subarray(i, i + len).toString('utf8'));
      at = buf.indexOf(LOCATION, at + 1);
    }
    tail = buf.subarray(Math.max(0, buf.length - 4096));
  }
  return found;
}

function graphs(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, e.name);
    if (e.isDirectory() && !(dir === out && e.name === 'export')) out.push(...graphs(path));
    else if (e.isFile() && e.name.endsWith('.onnx')) out.push(path);
  }
  return out;
}
report.externalData = {};
for (const graph of graphs(out)) {
  const rel = relative(out, graph);
  const locations = [...(await externalData(graph))].sort();
  report.externalData[rel] = locations;
  for (const loc of locations) {
    const path = resolve(dirname(graph), loc);
    if (relative(out, path).startsWith('..') || !existsSync(path) || !statSync(path).isFile())
      problems.push(`${rel} keeps weights in ${loc}, which is not in the export`);
  }
}
const graphCount = Object.keys(report.externalData).length;
say(`external data: ${graphCount} graph(s), ${Object.values(report.externalData).flat().length} data file reference(s) checked`);

if (spec.check === 'true' && problems.length > 0) say('skipping ortllama check: the listing already failed');
if (spec.check === 'true' && problems.length === 0) {
  const target = `${id}:${entry.defaultVariant}`;
  say(`ortllama check ${target}`);
  const check = ortllama('check', target, '--json');
  let parsed = null;
  try {
    parsed = JSON.parse(check.stdout);
  } catch {}
  report.check = parsed ?? { status: check.status, stdout: check.stdout, stderr: check.stderr };
  if (check.status !== 0)
    problems.push(`ortllama check ${target} exited ${check.status}: ${(parsed?.summary ?? check.stderr).trim().split('\n').at(-1)}`);
  else say(`check ok: loaded in ${Math.round(parsed.loadMs)} ms, request ${Math.round(parsed.requestMs)} ms`);
}
finish();
