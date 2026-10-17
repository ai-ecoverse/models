#!/usr/bin/env node
/**
 * Read recipes/<name>/recipe.yaml fields (small YAML subset, no dependency).
 *
 *   node scripts/read-recipe.mjs tiny-random-llama --json
 *   node scripts/read-recipe.mjs tiny-random-llama --field target.repo
 *   node scripts/read-recipe.mjs tiny-random-llama --env    # RECIPE_* shell assignments
 */
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const name = process.argv[2];
const mode = process.argv[3] || '--json';
const field = process.argv[4];

if (!name || name.startsWith('-')) {
  console.error(
    'usage: read-recipe.mjs <recipe> [--json|--env|--field <key>]'
  );
  process.exit(2);
}

const path = join(root, 'recipes', name, 'recipe.yaml');
if (!existsSync(path)) {
  console.error(`missing ${path}`);
  process.exit(1);
}

/**
 * Minimal YAML subset: 2-space indent, scalars, null, [], maps, `- item` lists.
 * Empty nested keys are maps; use `key: []` for empty lists.
 * @param {string} text
 */
function parseSimpleYaml(text) {
  /** @type {Record<string, unknown>} */
  const rootObj = {};
  /** @type {{ indent: number, container: Record<string, unknown> | unknown[] }[]} */
  const stack = [{ indent: -1, container: rootObj }];

  const unquote = (s) => {
    if (
      (s.startsWith('"') && s.endsWith('"')) ||
      (s.startsWith("'") && s.endsWith("'"))
    ) {
      return s.slice(1, -1);
    }
    return s;
  };

  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    const indent = raw.match(/^ */)[0].length;
    const line = raw.trim();

    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }
    const top = stack[stack.length - 1];

    if (line.startsWith('- ')) {
      if (!Array.isArray(top.container)) {
        // Promote an empty map created by `key:` into a list when the first
        // `- item` appears (Homebrew-style nested dependency lists).
        const emptyMap =
          top.container &&
          typeof top.container === 'object' &&
          !Array.isArray(top.container) &&
          Object.keys(top.container).length === 0;
        if (emptyMap && stack.length > 1) {
          const parentFrame = stack[stack.length - 2];
          const parent = parentFrame.container;
          if (parent && typeof parent === 'object' && !Array.isArray(parent)) {
            // Find which key points at this empty map and replace with [].
            const map = /** @type {Record<string, unknown>} */ (parent);
            for (const [k, v] of Object.entries(map)) {
              if (v === top.container) {
                const list = [];
                map[k] = list;
                top.container = list;
                break;
              }
            }
          }
        }
      }
      if (!Array.isArray(top.container)) {
        console.error(`list item under non-list at indent ${indent}: ${line}`);
        process.exit(1);
      }
      top.container.push(unquote(line.slice(2).trim()));
      continue;
    }

    const m = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!m) continue;
    if (Array.isArray(top.container)) {
      console.error(`map key under list: ${line}`);
      process.exit(1);
    }
    const key = m[1];
    const rest = m[2];
    const parent = /** @type {Record<string, unknown>} */ (top.container);

    if (rest === '[]') {
      parent[key] = [];
      continue;
    }
    if (rest === 'null' || rest === '~') {
      parent[key] = null;
      continue;
    }
    if (rest === '') {
      const child = {};
      parent[key] = child;
      stack.push({ indent, container: child });
      continue;
    }
    parent[key] = unquote(rest);
  }

  return rootObj;
}

const recipe = parseSimpleYaml(readFileSync(path, 'utf8'));

if (mode === '--json') {
  console.log(JSON.stringify(recipe, null, 2));
  process.exit(0);
}

if (mode === '--field') {
  if (!field) {
    console.error('--field requires a key');
    process.exit(2);
  }
  const parts = field.split('.');
  let cur = /** @type {unknown} */ (recipe);
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') {
      cur = undefined;
      break;
    }
    cur = /** @type {Record<string, unknown>} */ (cur)[p];
  }
  if (cur === undefined || cur === null) process.exit(0);
  if (typeof cur === 'object') console.log(JSON.stringify(cur));
  else console.log(String(cur));
  process.exit(0);
}

if (mode === '--env') {
  // Flatten scalars to RECIPE_<PATH>=<value>, lists joined by spaces, each
  // value single-quoted for `eval` in scripts/run.sh.
  /** @param {unknown} v @param {string} prefix */
  const walk = (v, prefix) => {
    if (v == null) return;
    if (Array.isArray(v)) {
      emit(prefix, v.join(' '));
    } else if (typeof v === 'object') {
      for (const [k, x] of Object.entries(v)) walk(x, `${prefix}_${k}`);
    } else {
      emit(prefix, String(v));
    }
  };
  /** @param {string} key @param {string} value */
  const emit = (key, value) => {
    const name = key.toUpperCase().replace(/[^A-Z0-9_]/g, '_');
    console.log(`${name}='${value.replace(/'/g, `'\\''`)}'`);
  };
  walk(recipe, 'RECIPE');
  process.exit(0);
}

console.error(`unknown mode ${mode}`);
process.exit(2);
