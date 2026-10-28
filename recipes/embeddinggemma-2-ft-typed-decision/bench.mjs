// Time one request of the golden fixtures on onnxruntime-node (CPU, WebGPU) or onnxruntime-web (WebGPU on Dawn,
// WASM), and check its probabilities against the fixture. Not run by convert.sh (CI has no GPU); for the numbers in
// the card. Uses the fixture's token ids and layout, so it needs no tokenizer or rendering port.
//
//   npm i onnxruntime-node@1.30.0 onnxruntime-web@1.30.0 webgpu@0.6.2
//   node bench.mjs <export dir> <variant> <node-cpu|node-webgpu|web-webgpu|web-wasm> [fixture name] [onnx dir]
import { readFileSync } from 'node:fs';
import { join, basename } from 'node:path';

const [dir, variant = 'fp32', backend = 'node-cpu', name = 'card-agent-trace', onnxDir] = process.argv.slice(2);
const SUFFIX = { fp32: '', q8: '_quantized', fp16: '_fp16', q4: '_q4', q4f16: '_q4f16', q8f16: '_q8f16' };
const fixture = JSON.parse(readFileSync(join(dir, 'fixtures/typed-decision.json'), 'utf8')).fixtures.find((f) => f.name === name);
const encPath = join(onnxDir ?? join(dir, 'onnx'), `model${SUFFIX[variant]}.onnx`);
const headPath = join(dir, 'onnx/decision_head.onnx');

let ort;
let opts;
if (backend.startsWith('node')) {
  ort = (await import('onnxruntime-node')).default;
  opts = { executionProviders: [backend === 'node-webgpu' ? 'webgpu' : 'cpu'] };
} else {
  if (backend === 'web-webgpu') {
    const { create, globals } = await import('webgpu');
    Object.assign(globalThis, globals);
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true });
    ort = await import('onnxruntime-web/webgpu');
  } else {
    ort = await import('onnxruntime-web');
  }
  opts = { executionProviders: [backend === 'web-webgpu' ? 'webgpu' : 'wasm'] };
}

async function session(path, withData) {
  if (backend.startsWith('node')) return ort.InferenceSession.create(path, opts);
  const o = { ...opts };
  if (withData) o.externalData = [{ path: basename(path) + '_data', data: readFileSync(path + '_data') }];
  return ort.InferenceSession.create(readFileSync(path), o);
}

const t0 = performance.now();
const enc = await session(encPath, true);
const head = await session(headPath, false);
const loadMs = performance.now() - t0;
const media = Object.fromEntries(
  enc.inputNames.filter((n) => n.endsWith('_features')).map((n) => [n, new ort.Tensor('float32', new Float32Array(0), [0, 512])])
);

async function embedBatch(ids) {
  const t = Math.max(...ids.map((i) => i.length));
  const input = new BigInt64Array(ids.length * t);
  const mask = new BigInt64Array(ids.length * t);
  ids.forEach((row, r) => row.forEach((id, c) => { input[r * t + c] = BigInt(id); mask[r * t + c] = 1n; }));
  const out = await enc.run({ input_ids: new ort.Tensor('int64', input, [ids.length, t]),
    attention_mask: new ort.Tensor('int64', mask, [ids.length, t]), ...media }, ['sentence_embedding']);
  return out.sentence_embedding.data;
}

async function embed(ids, slack = 1.25) {   // runtime.py's batching: similar lengths together
  const order = ids.map((_, i) => i).sort((a, b) => ids[a].length - ids[b].length);
  const out = new Float32Array(ids.length * 768);
  let batch = [], real = 0;
  for (const i of [...order, null]) {
    const n = i === null ? 0 : ids[i].length;
    if (batch.length && (i === null || (batch.length + 1) * n > slack * (real + n))) {
      const e = await embedBatch(batch.map((j) => ids[j]));
      batch.forEach((j, k) => out.set(e.subarray(k * 768, (k + 1) * 768), j * 768));
      batch = []; real = 0;
    }
    if (i !== null) { batch.push(i); real += n; }
  }
  return out;
}

const L = fixture.layout;
const i64 = (a, dims) => new ort.Tensor('int64', BigInt64Array.from(a.flat(), BigInt), dims);
async function decide() {
  const e = await embed(fixture.token_ids);
  const q = L.question_type.length, k = L.option_index[0].length;
  const out = await head.run({
    text_embeds: new ort.Tensor('float32', e, [fixture.token_ids.length, 768]),
    state_index: i64(L.state_index, [q]), question_index: i64(L.question_index, [q]),
    option_index: i64(L.option_index, [q, k]),
    option_mask: new ort.Tensor('bool', Uint8Array.from(L.option_mask.flat(), Number), [q, k]),
    question_type: i64(L.question_type, [q]),
  });
  return { probs: out.probabilities.data, k };
}

const t1 = performance.now();
const first = await decide();
const firstMs = performance.now() - t1;
const ms = [];
for (let r = 0; r < 20; r++) { const t = performance.now(); await decide(); ms.push(performance.now() - t); }
ms.sort((a, b) => a - b);
let dp = 0;
fixture.probabilities.forEach((p, qi) => p.forEach((x, oi) => { dp = Math.max(dp, Math.abs(x - first.probs[qi * first.k + oi])); }));
console.log(JSON.stringify({ variant, backend, fixture: name, texts: fixture.texts.length,
  tokens: fixture.token_ids.reduce((s, i) => s + i.length, 0), load_ms: Math.round(loadMs), first_ms: Math.round(firstMs),
  median_ms: +ms[10].toFixed(1), min_ms: +ms[0].toFixed(1), max_abs_dp_vs_fixture: dp }));
process.exit(0);
