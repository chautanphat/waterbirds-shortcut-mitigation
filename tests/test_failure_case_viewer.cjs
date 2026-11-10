/* Run with node --test tests/test_failure_case_viewer.cjs. No packages required. */
'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../scripts/failure_case_viewer.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const sandbox = {};
vm.runInNewContext(script, sandbox, { filename:'failure_case_viewer.html' });
const F = sandbox.FailureCases;
const plain = value => JSON.parse(JSON.stringify(value));
function row(path, truth, pred, background, extra = {}) {
  return { image_path:path, true_label:String(truth), predicted_label:String(pred),
    spurious_attribute:String(background), split:'test', ...extra };
}
function run(rows, id, opts = {}) {
  const result = F.normalizeRows(rows, { scope:'predictions', ...opts })[0];
  result.id = id;
  return result;
}

test('CSV reads BOM, CRLF, quoted commas, escaped quotes, and embedded newlines', () => {
  const records = F.parseCSV('\uFEFFimage_path,note\r\n"species/a,b.jpg","line one\nline ""two"""\r\n');
  assert.deepEqual(plain(records), [{ image_path:'species/a,b.jpg', note:'line one\nline "two"' }]);
});

test('malformed CSV fails explicitly instead of silently dropping columns or records', () => {
  for (const csv of ['a,a\n1,2', 'a,b\n1', 'a\n"unclosed', 'a\n"x"trailing', 'a\nfoo"bar', 'a,b\n']) {
    assert.throws(() => F.parseCSV(csv));
  }
  assert.deepEqual(plain(F.parseCSV('a,b\n1,2\n\n')), [{ a:'1', b:'2' }]);
});

test('existing AutoBackSwap export is accepted with one-based groups and seeds', () => {
  const csv = 'image_path,true_label,predicted_label,correct,group,group_name,spurious_attribute,confidence,run_label,seed\n' +
    '001.Albatross/a.jpg,waterbird,landbird,0,3,Waterbird on Land,land,0.96,ours_minority_false_seed0,0\n' +
    '002.Sparrow/b.jpg,landbird,landbird,1,1,Landbird on Land,land,0.82,ours_minority_false_seed0,0\n';
  const result = F.normalizeRows(F.parseCSV(csv), { filename:'predictions.csv', split:'test' })[0];
  assert.equal(result.scope, 'predictions');
  assert.match(result.label, /seed 0/);
  assert.deepEqual([...result.rows.values()].map(r => r.group), [2, 0]);
  assert.equal([...result.rows.values()][0].split, 'test');
});

test('zero-based group-only exports work only under the selected convention', () => {
  const records = [{ filename:'bird/a.jpg', y:'1', y_pred:'0', group:'2' }];
  const result = F.normalizeRows(records, { groupBase:0 })[0];
  assert.equal([...result.rows.values()][0].background, 0);
  assert.throws(() => F.normalizeRows(records, { groupBase:1 }), /Group conflicts/);
});

test('contradictory labels, groups, correctness, and invalid confidence are rejected', () => {
  const invalid = [
    row('a.jpg', 0, 1, 0, { correct:'1' }),
    row('a.jpg', 0, 1, 0, { correct:'yes' }),
    row('a.jpg', 0, 1, 0, { group:'3' }),
    row('a.jpg', 0, 1, 0, { confidence:'NaN' }),
    row('a.jpg', 0, 1, 0, { confidence:'90' }),
    row('a.jpg', 0, 1, 0, { confidence:'Infinity' }),
    row('a.jpg', 0, 1, 0, { true_label:'2' }),
    row('a.jpg', 0, 1, 0, { split:'development' }),
    { image_path:'a.jpg', true_label:'0', predicted_label:'1' },
  ];
  invalid.forEach(r => assert.throws(() => F.normalizeRows([r])));
});

test('relative image paths normalize Windows separators and reject URLs and traversal', () => {
  assert.equal(F.relativePath('.\\birds\\one.jpg'), 'birds/one.jpg');
  for (const p of ['', '/root/a.jpg', 'C:\\data\\a.jpg', 'https://example.test/a.jpg', '../a.jpg', 'birds/../a.jpg', 'birds//a.jpg', 'birds/./a.jpg']) {
    assert.throws(() => F.relativePath(p));
  }
});

test('duplicate rows fail within a run; seeds identify separate runs', () => {
  const r = row('bird/a.jpg', 0, 1, 1, { seed:'0', run_label:'ERM' });
  assert.throws(() => F.normalizeRows([r, r]), /Duplicate image/);
  assert.equal(F.normalizeRows([r, { ...r, seed:'1' }]).length, 2);
});

test('errors-only exports never produce fabricated accuracy or WGA', () => {
  const rows = [row('a.jpg', 0, 1, 0), row('b.jpg', 0, 1, 1), row('c.jpg', 1, 0, 0), row('d.jpg', 1, 0, 1)];
  const inferred = F.normalizeRows(rows, { filename:'predictions.csv' })[0];
  assert.equal(inferred.scope, 'errors');
  const m = F.metrics(inferred);
  assert.equal(m.count, 4); assert.equal(m.errors, 4); assert.equal(m.rate, null); assert.equal(m.worst, null);
  const explicit = run(rows, 'full');
  assert.equal(F.metrics(explicit).rate, 0);
  assert.equal(F.metrics(explicit).worst, 0);
  assert.throws(() => F.normalizeRows([row('a.jpg', 0, 0, 0)], { scope:'errors' }), /contains correct/);
});

test('accuracy and WGA use actual loaded denominators and the requested split', () => {
  const full = run([
    row('a.jpg', 0, 0, 0), row('b.jpg', 0, 1, 1), row('c.jpg', 1, 1, 0), row('d.jpg', 1, 1, 1),
    row('e.jpg', 0, 0, 1, { split:'val' }),
  ], 'full');
  const m = F.metrics(full, 'test');
  assert.equal(m.count, 4); assert.equal(m.errors, 1); assert.equal(m.rate, .75); assert.equal(m.worst, 0);
  assert.deepEqual(plain(m.groups), [{ count:1, errors:0 }, { count:1, errors:1 }, { count:1, errors:0 }, { count:1, errors:0 }]);
  assert.equal(F.metrics(full, 'val').worst, null);
  assert.equal(F.metrics(full, 'train').rate, null);
});

test('matching uses path and split and rejects conflicting ground truth across runs', () => {
  const a = run([row('bird/a.jpg', 0, 1, 0)], 'a');
  const b = run([row('bird/a.jpg', 0, 0, 0, { split:'val' })], 'b');
  assert.equal(F.mergeExamples([a, b]).length, 2);
  assert.equal(F.mergeExamples([a, b], 'test').length, 1);
  const conflict = run([row('bird/a.jpg', 1, 1, 0)], 'bad');
  assert.throws(() => F.mergeExamples([a, conflict]), /Conflicting ground truth/);
});

test('comparison modes preserve missing predictions as unknown', () => {
  const a = run([row('both.jpg', 0, 1, 0), row('fixed.jpg', 1, 0, 0), row('only-a.jpg', 0, 1, 1), row('regressed.jpg', 1, 1, 1)], 'a');
  const b = run([row('both.jpg', 0, 1, 0), row('fixed.jpg', 1, 1, 0), row('only-b.jpg', 1, 0, 1), row('regressed.jpg', 1, 0, 1)], 'b');
  const examples = F.mergeExamples([a, b]);
  const matching = mode => plain(examples.filter(e => F.matchesMode(e, 'a', 'b', mode)).map(e => e.path).sort());
  assert.deepEqual(matching('a-wrong-b-right'), ['fixed.jpg']);
  assert.deepEqual(matching('a-right-b-wrong'), ['regressed.jpg']);
  assert.deepEqual(matching('both-wrong'), ['both.jpg']);
  assert.deepEqual(matching('disagree'), ['fixed.jpg', 'regressed.jpg']);
  assert.equal(matching('any-wrong').length, 5);
  assert.equal(matching('a-wrong').length, 3);
  assert.equal(examples.filter(e => F.matchesMode(e, 'a', '', 'a-wrong-b-right')).length, 0);
});

test('search, group and confidence filters combine; missing confidence sorts last', () => {
  const a = run([
    row('sparrow/a.jpg', 0, 1, 1, { confidence:'.9' }),
    row('sparrow/b.jpg', 0, 1, 0, { confidence:'.99' }),
    row('albatross/c.jpg', 1, 0, 0, { confidence:'.7' }),
    row('sparrow/d.jpg', 0, 1, 1),
  ], 'a');
  const examples = F.mergeExamples([a]);
  const opts = { a:'a', b:'', mode:'a-wrong', group:1, search:'SPARROW', minConfidence:.8, sort:'confidence-desc' };
  assert.deepEqual(plain(F.filterExamples(examples, opts).map(e => e.path)), ['sparrow/a.jpg']);
  const ascending = F.filterExamples(examples, { ...opts, group:null, search:'', minConfidence:0, sort:'confidence-asc' });
  assert.deepEqual(plain(ascending.map(e => e.path)), ['albatross/c.jpg', 'sparrow/a.jpg', 'sparrow/b.jpg', 'sparrow/d.jpg']);
});

test('folder matching supports original and images/ layouts without ambiguous basename matches', () => {
  const a = { name:'a.jpg', webkitRelativePath:'waterbirds/images/001.Bird/a.jpg' };
  const b = { name:'a.jpg', webkitRelativePath:'waterbirds/images/002.Bird/a.jpg' };
  const c = { name:'meta.csv', webkitRelativePath:'waterbirds/meta.csv' };
  const index = F.imageIndex([a, b, c]);
  assert.equal(index.get('001.Bird/a.jpg'), a);
  assert.equal(index.get('images/001.Bird/a.jpg'), a);
  assert.equal(index.get('a.jpg'), null);
  assert.equal(index.has('meta.csv'), false);
});

test('export retains source, seed, unknown predictions and protects spreadsheet formula cells', () => {
  const a = run([row('=bad.jpg', 0, 1, 1, { run_label:'ERM', seed:'0', confidence:'.9' })], 'a', { filename:'first.csv' });
  const b = run([row('other.jpg', 0, 0, 0)], 'b', { filename:'second.csv' });
  const csv = F.exportCSV(F.mergeExamples([a, b]), [a, b]);
  const records = F.parseCSV(csv);
  assert.equal(records.length, 4);
  assert.equal(records[0].image_path, "'=bad.jpg");
  assert.equal(records[0].source_csv, 'first.csv');
  assert.equal(records[0].seed, '0');
  assert.equal(records[1].predicted_label, '');
  assert.equal(records[1].correct, '');
});

test('all input data is rendered as text; app has no external scripts, dependencies or requests', () => {
  assert.doesNotMatch(script, /\.innerHTML\s*=|\beval\s*\(|\bfetch\s*\(|XMLHttpRequest/);
  assert.doesNotMatch(html, /<script[^>]+src=|<link[^>]+href=/);
  assert.match(html, /default-src 'none'/);
});

// A small DOM substitute checks event wiring and rendering without extra packages.
// It does not check browser layout, image decoding, or the native folder picker.
function mountViewer() {
  class Element {
    constructor(tag = 'div') { this.tag = tag; this.children = []; this.handlers = {}; this._value = ''; this._text = ''; this.hidden = false; this.disabled = false; this.open = false; this.style = {}; this.attributes = {}; }
    append(...elements) { for (const child of elements) { child.parent = this; this.children.push(child); } }
    replaceChildren(...elements) { this.children = []; this._text = ''; this._value = ''; this.append(...elements); }
    replaceWith(element) { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1, element); }
    remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(name, handler) { this.handlers[name] = handler; }
    get firstChild() { return this.children[0]; }
    get lastChild() { return this.children.at(-1); }
    get value() { return this._value; }
    set value(value) { this._value = value; }
    get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
    set textContent(value) { this._text = value; this.children = []; }
    showModal() { this.open = true; }
    close() { this.open = false; }
    click() { this.onclick?.(); }
    getContext() { return new Proxy({}, { get:() => () => {} }); }
    toDataURL() { return 'data:image/png;base64,aW1hZ2U='; }
  }
  const ids = new Map();
  for (const match of html.matchAll(/<([a-z-]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = new Element(match[1]); element.hidden = /\bhidden\b/.test(match[0]); ids.set(match[2], element);
  }
  for (const match of html.matchAll(/<select id="([^"]+)">([\s\S]*?)<\/select>/g)) {
    const first = match[2].match(/<option value="([^"]*)"/); if (first) ids.get(match[1]).value = first[1];
  }
  const document = { getElementById:id => { if (!ids.has(id)) throw new Error(`Missing element ${id}`); return ids.get(id); }, createElement:tag => new Element(tag), body:new Element('body') };
  let nextURL = 0;
  const urls = { createObjectURL:() => `blob:local-test-${++nextURL}`, revokeObjectURL:() => {} };
  const context = { document, URL:urls, Blob, Uint8Array, atob, setTimeout:fn => fn() };
  vm.runInNewContext(script, context, { filename:'failure_case_viewer.html' });
  return id => document.getElementById(id);
}

test('UI demo renders, A/B filters and group buttons work, dialog navigates, reset clears filters', () => {
  const $ = mountViewer();
  $('demo').click();
  assert.equal($('demo-banner').hidden, false);
  assert.equal($('run-summary').children.length, 2);
  assert.equal($('gallery').children.length, 6);
  assert.equal($('run-a').value, 'run-1'); assert.equal($('run-b').value, 'run-2');
  $('mode').value = 'a-wrong-b-right'; $('mode').onchange();
  assert.equal($('gallery').children.length, 4);
  $('groups').children[3].click();
  assert.equal($('gallery').children.length, 1);
  $('gallery').firstChild.firstChild.click();
  assert.equal($('detail').open, true);
  assert.equal($('detail-predictions').children.length, 2);
  assert.match($('detail-path').textContent, /example_4\.png/);
  $('close-detail').click(); assert.equal($('detail').open, false);
  $('reset').click(); assert.equal($('gallery').children.length, 6);
  $('gallery').firstChild.firstChild.click();
  const before = $('detail-path').textContent;
  $('detail-next').click(); assert.notEqual($('detail-path').textContent, before);
  $('detail').handlers.keydown({ key:'ArrowLeft', preventDefault:() => {} });
  assert.equal($('detail-path').textContent, before);
  $('export').click();
  $('clear').click();
  assert.equal($('workspace').hidden, true); assert.equal($('welcome').hidden, false);
});

test('UI imports files, keeps valid files after another fails, and clears synthetic runs', async () => {
  const $ = mountViewer(); $('demo').click();
  const good = { name:'predictions.csv', text:async () => 'image_path,true_label,predicted_label,spurious_attribute,run_label\nbird/a.jpg,0,1,0,Real ERM\nbird/b.jpg,0,0,0,Real ERM' };
  const bad = { name:'bad.csv', text:async () => 'overall_accuracy,wga\n0.9,0.8' };
  const target = { files:[good, bad], value:'selected', disabled:false };
  await $('csv-files').handlers.change({ target });
  assert.equal($('demo-banner').hidden, true);
  assert.equal($('run-summary').children.length, 1);
  assert.match($('run-summary').textContent, /Real ERM/);
  assert.match($('notice').textContent, /bad\.csv/);
  assert.equal(target.disabled, false); assert.equal(target.value, '');
  const conflict = { name:'conflict.csv', text:async () => 'image_path,true_label,predicted_label,spurious_attribute\nbird/a.jpg,1,1,0' };
  await $('csv-files').handlers.change({ target:{ files:[conflict] } });
  assert.equal($('run-summary').children.length, 1);
  assert.match($('notice').textContent, /Conflicting ground truth/);
});

test('UI paginates more than 24 failures and applies the confidence threshold', async () => {
  const $ = mountViewer();
  const data = 'image_path,true_label,predicted_label,spurious_attribute,confidence\n' + Array.from({ length:30 }, (_, i) => `bird/${i}.jpg,0,1,0,${i < 10 ? .95 : .7}`).join('\n');
  await $('csv-files').handlers.change({ target:{ files:[{ name:'errors.csv', text:async () => data }] } });
  assert.equal($('gallery').children.length, 24);
  assert.equal($('previous').disabled, true); assert.equal($('next').disabled, false);
  $('next').click(); assert.equal($('gallery').children.length, 6);
  $('min-confidence').value = '.9'; $('min-confidence').onchange();
  assert.equal($('gallery').children.length, 10);
  assert.equal($('previous').disabled, true); assert.equal($('next').disabled, true);
  $('search').value = 'no-such-species'; $('search').oninput();
  assert.match($('gallery').textContent, /No matching cases/);
  assert.equal($('export').disabled, true);
});

test('experiment folder loads predictions and images, skips metadata and redundant errors, and selects B', async () => {
  const $ = mountViewer();
  const csv = (relativePath, label) => ({ name:relativePath.split('/').at(-1), webkitRelativePath:relativePath,
    text:async () => `image_path,true_label,predicted_label,spurious_attribute,run_label\nbird/a.jpg,0,1,0,${label}\nbird/b.jpg,0,0,0,${label}` });
  const files = [
    csv('experiment/erm/predictions.csv', 'ERM'),
    { name:'errors.csv', webkitRelativePath:'experiment/erm/errors.csv', text:async () => { throw new Error('redundant errors should not be read'); } },
    csv('experiment/approach_b/predictions.csv', 'Approach B'),
    { name:'metadata.csv', webkitRelativePath:'experiment/images/metadata.csv', text:async () => 'img_filename,y,place\nbird/a.jpg,0,0' },
    { name:'a.jpg', webkitRelativePath:'experiment/images/bird/a.jpg' },
    { name:'b.jpg', webkitRelativePath:'experiment/images/bird/b.jpg' },
  ];
  await $('experiment-folder').handlers.change({ target:{ files } });
  assert.equal($('run-summary').children.length, 2);
  assert.equal($('run-b').value, 'run-2');
  assert.match($('notice').textContent, /Skipped 2/);
  assert.match($('image-status').textContent, /2 image files/);
  assert.doesNotMatch($('count').textContent, /images missing/);
  assert.match($('setup-status').textContent, /2 run\(s\) loaded/);
  assert.equal($('experiment-folder').disabled, false);
});

test('experiment folder without predictions explains what is missing and sample CSV downloads', async () => {
  const $ = mountViewer();
  await $('experiment-folder').handlers.change({ target:{ files:[{ name:'metadata.csv', text:async () => 'img_filename,y,place\nbird/a.jpg,0,0' }] } });
  assert.match($('notice').textContent, /No prediction CSVs found/);
  assert.equal($('workspace').hidden, true);
  $('sample').click();
});
