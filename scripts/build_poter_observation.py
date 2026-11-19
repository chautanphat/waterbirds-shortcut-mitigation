"""Build an offline observation gallery; preserve uncertainty of imported labels."""
import json
from pathlib import Path
from html import escape
from collections import Counter
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=Path, default=Path('D:/POTER_data/cub/data/waterbird_complete95_forest2water2/metadata.csv'))
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/poter/visualizer'
data = json.loads((BASE / 'error_samples_summary.json').read_text(encoding='utf-8'))
cards = []
for i, row in enumerate(data['errors']):
    path = row['path'].replace('\\', '/')
    image = BASE / 'images' / path
    if not image.is_file() or not image.resolve().is_relative_to((BASE / 'images').resolve()):
        raise ValueError(f'Missing or invalid image: {path}')
    cards.append(f'''<article data-group="{escape(row['group'], quote=True)}">
<img loading="lazy" src="images/{escape(path, quote=True)}" alt="{escape(row['species'])}">
<h3>#{i+1} {escape(row['species'])}</h3>
<p>{escape(row['group'])}</p>
<p>Nhãn trong danh sách: {escape(row['true'])} → {escape(row['pred'])}</p>
<small>{escape(path)}</small>
<p><a href="images/{escape(path, quote=True)}" target="_blank">Mở ảnh gốc</a></p>
<textarea data-key="{escape(path, quote=True)}" placeholder="Quan sát: chim nhỏ? bị che? nền nước/đất? nhãn có hợp lý?" aria-label="Ghi chú ảnh {i+1}"></textarea></article>''')
groups = sorted({r['group'] for r in data['errors']})
counts = Counter(r['species'] for r in data['errors'])
ranking = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
with args.metadata.open(encoding='utf-8-sig', newline='') as stream:
    test_rows = [r for r in csv.DictReader(stream) if str(r['split']) == '2']
test_paths = {r['img_filename'].replace('\\', '/') for r in test_rows}
error_paths = [r['path'].replace('\\', '/') for r in data['errors']]
assert len(test_rows) == 5794, 'Expected official 5794-image test split'
assert len(error_paths) == len(set(error_paths)), 'Duplicate errors'
assert set(error_paths) <= test_paths, 'Error images outside test split'
totals = Counter(r['img_filename'].split('/')[0] for r in test_rows)
errors_by_species = Counter(p.split('/')[0] for p in error_paths)
ranking_rows = sorted(((key.split('.', 1)[1].replace('_', ' '), errors_by_species[key], total) for key, total in totals.items()), key=lambda r: (-r[1]/r[2], -r[1], r[0]))
table = '<h2>Tỷ lệ sai theo loài chim — xếp hạng giảm dần</h2><p>Số ảnh trong danh sách lỗi / tổng ảnh TEST của đúng loài × 100. Mẫu số lấy từ metadata Waterbirds, split=2 (5.794 ảnh). Tử số vẫn là danh sách lỗi chưa xác minh; 0 nghĩa là không có ảnh trong danh sách này.</p><table><thead><tr><th>Hạng</th><th>Loài chim</th><th>Số ảnh sai*</th><th>Tổng ảnh test</th><th>Tỷ lệ sai*</th></tr></thead><tbody>'
table += ''.join(f'<tr><td>{i}</td><td>{escape(name)}</td><td>{n}</td><td>{total}</td><td>{100*n/total:.2f}%</td></tr>' for i, (name, n, total) in enumerate(ranking_rows, 1))
table += '</tbody></table>'
report = '# Tỷ lệ sai theo loài chim\n\nMẫu số: metadata Waterbirds split=2, 5.794 ảnh test. Tử số: danh sách lỗi chưa xác minh bằng checkpoint/dự đoán gốc. Xếp theo tỷ lệ giảm dần; 0 là không xuất hiện trong danh sách lỗi.\n\n| Hạng | Loài | Số ảnh sai* | Tổng ảnh test | Tỷ lệ sai* |\n| ---: | --- | ---: | ---: | ---: |\n'
report += '\n'.join(f'| {i} | {name} | {n} | {total} | {100*n/total:.2f}% |' for i, (name, n, total) in enumerate(ranking_rows, 1))
print('Test denominator:', len(test_rows), 'species:', len(totals), 'listed errors:', sum(errors_by_species.values()))
print('Top 3:', ranking_rows[:3])
"""Legacy share-of-errors table, replaced by per-species test rates above.
table = '<h2>Xếp hạng theo loài chim</h2><p>Tỷ lệ là số ảnh của loài / tổng số ảnh trong danh sách lỗi chưa xác minh; không phải tỷ lệ sai trên toàn bộ ảnh test của loài.</p><table><thead><tr><th>Hạng</th><th>Loài chim</th><th>Số ảnh</th><th>% danh sách lỗi</th></tr></thead><tbody>'
table += ''.join(f'<tr><td>{i}</td><td>{escape(name)}</td><td>{n}</td><td>{100*n/len(data["errors"]):.2f}%</td></tr>' for i, (name, n) in enumerate(ranking, 1))
table += '</tbody></table>'
report = '# Thống kê ảnh theo loài chim\n\nNguồn: `error_samples_summary.json`; danh sách dự đoán sai chưa được xác minh bằng checkpoint/dự đoán gốc.\n\nPhần trăm = số ảnh của loài / tổng số ảnh trong danh sách. Chưa có tổng số ảnh test mỗi loài nên chưa tính được tỷ lệ sai mỗi loài. Chỉ liệt kê các loài xuất hiện trong danh sách; không suy ra các loài còn lại có 0 lỗi. Đồng số ảnh được sắp theo tên.\n\n| Hạng | Loài | Số ảnh | % danh sách lỗi |\n| ---: | --- | ---: | ---: |\n'
report += '\n'.join(f'| {i} | {name} | {n} | {100*n/len(data["errors"]):.2f}% |' for i, (name, n) in enumerate(ranking, 1))
"""
(BASE / 'species_ranking.md').write_text(report + '\n', encoding='utf-8')
options = ''.join(f'<option>{escape(g)}</option>' for g in groups)
page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>POTER — Quan sát ảnh</title>
<style>body{font:16px system-ui;background:#f4f6f8;color:#18232d;margin:24px}header{max-width:1000px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}article{background:white;padding:16px;border-radius:12px}img{width:100%;height:250px;object-fit:contain;background:#eee}textarea{box-sizing:border-box;width:100%;min-height:90px}small{overflow-wrap:anywhere}button,select{padding:10px;margin:12px 8px 18px 0}.notice{background:#fff0c2;padding:16px;border-radius:8px}</style>
<header><h1>POTER — Bộ ảnh để quan sát lỗi</h1>
<p class="notice">346 ảnh từ danh sách có sẵn, <b>chưa xác minh bằng dự đoán gốc/checkpoint</b>. Nhãn thật và nhãn dự đoán hiển thị là metadata nhập lại. Không dùng bộ này để tính error rate hoặc kết luận nguyên nhân. Confidence, ERM comparison và lý do lỗi có sẵn được bỏ khỏi trang vì chưa có bằng chứng kiểm chứng.</p>
<p>Quan sát ảnh gốc, ghi nhận xét rồi xuất ghi chú. Nền ảnh có thể gợi ý shortcut nhưng không chứng minh nguyên nhân model đoán sai.</p>
RANKING
<select id="filter"><option value="all">Tất cả nhóm</option>OPTIONS</select>
<button id="export">Xuất ghi chú JSON</button><span id="count"></span></header>
<div id="grid">CARDS</div><script>
const cards=[...document.querySelectorAll('article')];
filter.onchange=()=>{let n=0;cards.forEach(c=>{c.hidden=filter.value!=='all'&&c.dataset.group!==filter.value;if(!c.hidden)n++});count.textContent=n+' ảnh'};filter.onchange();
let notes={};try{notes=JSON.parse(localStorage.getItem('poter-observation-v1')||'{}')}catch(e){}
document.querySelectorAll('textarea').forEach(t=>{t.value=notes[t.dataset.key]||'';t.oninput=()=>{notes[t.dataset.key]=t.value;try{localStorage.setItem('poter-observation-v1',JSON.stringify(notes))}catch(e){}}});
document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({status:'unverified_imported_error_list',notes},null,2)],{type:'application/json'}));a.download='poter-observations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''.replace('RANKING', table).replace('OPTIONS', options).replace('CARDS', '\n'.join(cards))
(BASE / 'observation.html').write_text(page, encoding='utf-8')
print(f'Created observation.html with {len(cards)} verified local image paths; prediction labels remain unverified.')
