import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    ortllama: { type: 'string' },
    tokenizer: { type: 'string' },
    ref: { type: 'string', multiple: true },
    results: { type: 'string' },
  },
});

const pkg = join(values.ortllama, 'node_modules', '@ai-ecoverse', 'slicc-ortllama');
const load = (rel) => import(pathToFileURL(join(pkg, rel)).href);
const { normalizeEnglish, plan } = await load('src/phonon/text.js');
const { createTokenizer } = await load('src/phonon/tokenizer.js');
const { version } = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8'));
const tokenizer = createTokenizer(JSON.parse(readFileSync(values.tokenizer, 'utf8')));

const report = { ortllama: version, sets: {}, mismatches: [] };
for (const file of values.ref) {
  const rows = readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  let same = 0;
  let tokens = 0;
  for (const ref of rows) {
    const normalized = normalizeEnglish(ref.input);
    const got = plan(tokenizer, ref.input, { frameRate: 12.5 }).map((c) => ({
      text: c.text,
      tokens: c.tokens,
      frames_after_eos: c.framesAfterEos,
      frame_budget: c.frameBudget,
    }));
    const want = ref.chunks.map((c) => ({
      text: c.text,
      tokens: c.tokens,
      frames_after_eos: c.frames_after_eos,
      frame_budget: c.frame_budget,
    }));
    const ok = normalized === ref.normalized && JSON.stringify(got) === JSON.stringify(want);
    if (ok) {
      same++;
      tokens += want.reduce((n, c) => n + c.tokens.length, 0);
    } else if (report.mismatches.length < 50) {
      report.mismatches.push({ set: file, input: ref.input, normalized, want: ref.normalized, got, chunks: want });
    }
  }
  report.sets[file.split('/').at(-1)] = { inputs: rows.length, identical: same, tokens };
}
report.ok = Object.values(report.sets).every((s) => s.inputs > 0 && s.identical === s.inputs);
writeFileSync(values.results, `${JSON.stringify(report, null, 1)}\n`);
for (const [name, s] of Object.entries(report.sets))
  console.log(`tokens: ${name}: ${s.identical}/${s.inputs} inputs identical to ptts (${s.tokens} tokens)`);
for (const m of report.mismatches.slice(0, 5)) console.log(`tokens: FAIL ${JSON.stringify(m.input)}`);
console.log(`tokens: slicc-ortllama ${version} ${report.ok ? 'passed' : 'failed'}`);
process.exit(report.ok ? 0 : 1);
