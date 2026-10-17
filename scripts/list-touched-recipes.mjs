#!/usr/bin/env node
/**
 * List recipe names touched between two git refs (or from a file list).
 *
 *   node scripts/list-touched-recipes.mjs --base origin/main --head HEAD [--json]
 *   echo 'recipes/tiny-random-llama/recipe.yaml' | node scripts/list-touched-recipes.mjs --stdin
 *
 * Prints one recipe name per line (or a JSON array with --json). Recipes with
 * `runner: retired` are left out. Exits 0 even when empty (caller decides).
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  /** @type {{ base?: string, head?: string, stdin: boolean, json: boolean }} */
  const out = { stdin: false, json: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--base') out.base = argv[++i];
    else if (a === '--head') out.head = argv[++i];
    else if (a === '--stdin') out.stdin = true;
    else if (a === '--json') out.json = true;
    else {
      console.error('usage: list-touched-recipes.mjs (--base <ref> --head <ref>) | --stdin [--json]');
      process.exit(2);
    }
  }
  return out;
}

/** @param {string[]} paths */
function recipesFromPaths(paths) {
  const set = new Set();
  for (const p of paths) {
    const m = p.replace(/\\/g, '/').match(/^recipes\/([^/]+)\//);
    if (!m) continue;
    const name = m[1];
    if (!existsSync(join(root, 'recipes', name, 'recipe.yaml'))) continue;
    const runner = spawnSync('node', [join(root, 'scripts/read-recipe.mjs'), name, '--field', 'runner'], {
      encoding: 'utf8',
      cwd: root,
    });
    if ((runner.stdout || '').trim() === 'retired') continue;
    set.add(name);
  }
  return [...set].sort();
}

function gitDiffNames(base, head) {
  const r = spawnSync('git', ['diff', '--name-only', `${base}...${head}`], { encoding: 'utf8', cwd: root });
  if (r.status !== 0) {
    console.error(r.stderr || r.stdout || 'git diff failed');
    process.exit(1);
  }
  return r.stdout.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

const args = parseArgs(process.argv);
let paths = [];
if (args.stdin) {
  paths = readFileSync(0, 'utf8').split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
} else if (args.base && args.head) {
  paths = gitDiffNames(args.base, args.head);
} else {
  console.error('need --base/--head or --stdin');
  process.exit(2);
}

const recipes = recipesFromPaths(paths);
if (args.json) console.log(JSON.stringify(recipes));
else for (const r of recipes) console.log(r);
