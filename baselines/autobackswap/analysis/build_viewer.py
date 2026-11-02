"""Build a single-file interactive viewer (viewer.html) for the misclassified test examples.

Reads errors.csv (from dump_predictions.py) and the dataset metadata.csv (test split only, for
per-species and per-scene totals). Writes viewer.html next to errors.csv. Thumbnails are loaded
from images/<image_path> next to the html (the folder from the local zip); nothing is uploaded.
"""
import argparse
import csv
import json
import os


def scene_of(place_filename):
    # "/o/ocean/00002178.jpg" -> "ocean"; "/l/lake/natural/0000.jpg" -> "lake/natural"
    parts = place_filename.strip("/").split("/")[1:-1]
    return "/".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--errors", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    meta = {}
    totals = {"species": {}, "scene": {}}
    with open(args.metadata) as f:
        for r in csv.DictReader(f):
            if r["split"] != "2":
                continue
            sp = r["img_filename"].split("/")[0]
            sc = scene_of(r["place_filename"])
            meta[r["img_filename"]] = {"scene": sc}
            g = int(r["y"]) * 2 + int(r["place"])  # 0-based group index
            for k, v in (("species", sp), ("scene", sc)):
                totals[k].setdefault(v, [0, 0, 0, 0])[g] += 1

    rows = []
    with open(args.errors) as f:
        for r in csv.DictReader(f):
            rows.append({
                "p": r["image_path"],
                "sp": r["image_path"].split("/")[0],
                "sc": meta[r["image_path"]]["scene"],
                "g": int(r["group"]),
                "gn": r["group_name"],
                "t": r["true_label"],
                "d": r["predicted_label"],
                "c": float(r["confidence"]),
            })

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.errors)), "viewer.html")
    html = TEMPLATE.replace("/*DATA*/", json.dumps({"rows": rows, "totals": totals}))
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, len(rows), "errors")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Error Explorer</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e0d8;--acc:#2f6f6a;--bad:#b5452f;--bar:#c9d9d6}
@media (prefers-color-scheme:dark){:root{--bg:#161715;--fg:#ecebe6;--mut:#9a9a92;--card:#1f201e;--line:#34352f;--acc:#6cb8b0;--bad:#e0836c;--bar:#2c4440}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}.sub{color:var(--mut);margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{font-size:22px;display:block}.card span{color:var(--mut);font-size:12px}
.card.sel{outline:2px solid var(--acc)}.card{cursor:pointer}
.bars .row{display:grid;grid-template-columns:190px 1fr 90px;gap:8px;align-items:center;padding:2px 0;cursor:pointer}
.bars .row:hover,.bars .row.sel{background:var(--card)}
.bars .lab{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bars .tr{background:var(--bar);height:12px;border-radius:2px;position:relative}
.bars .fi{background:var(--bad);height:12px;border-radius:2px;position:absolute;left:0;top:0}
.bars .n{color:var(--mut);text-align:right;font-variant-numeric:tabular-nums}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
.ctl select,.ctl input[type=text]{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
.chip{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:2px 9px;margin-right:6px;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.it{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:11px}
.it img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:var(--line);cursor:zoom-in}
.it div{padding:5px 7px;color:var(--mut)}.it b{color:var(--fg);font-weight:600}
.hist{display:flex;align-items:flex-end;gap:3px;height:70px}.hist i{flex:1;background:var(--bad);opacity:.8;min-height:1px}
.hl{display:flex;gap:3px;color:var(--mut);font-size:10px}.hl span{flex:1;text-align:center}
#big{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;flex-direction:column;color:#fff;z-index:9}
#big img{max-width:92vw;max-height:80vh}
.note{color:var(--mut);font-size:12px}
@media (max-width:600px){.bars .row{grid-template-columns:110px 1fr 70px}}
</style></head><body><main>
<h1>Error Explorer</h1>
<p class="sub">AutoBackSwap, Waterbirds without minority, seed 0 &middot; <span id="tot"></span>. Click a group card, species row or scene row to filter.</p>
<div class="cards" id="cards"></div>

<h2>Confidence of wrong predictions</h2>
<div class="hist" id="hist"></div><div class="hl" id="hl"></div>

<h2>Background scene (Places category) &mdash; errors / test images</h2>
<p class="note">Red = share of that scene's test images that are misclassified (within the selected group). Scenes with fewer than 5 test images are hidden.</p>
<div class="bars" id="scenes"></div>

<h2>Species &mdash; errors / test images</h2>
<p class="note">Sorted by error rate within the current group filter; species with fewer than 5 test images hidden.</p>
<div class="bars" id="species"></div>

<h2>Examples</h2>
<div class="ctl">
 <span id="chips"></span>
 <label>Sort <select id="sort"><option value="chi">confidence high to low</option><option value="clo">confidence low to high</option><option value="sp">species</option><option value="sc">scene</option></select></label>
 <label>Min confidence <input type="range" id="minc" min="0.5" max="1" step="0.01" value="0.5"> <span id="mincv">0.50</span></label>
 <label>Show <select id="lim"><option>60</option><option>120</option><option>400</option></select></label>
 <button id="clr">Clear filters</button>
</div>
<div class="grid" id="grid"></div>
<p class="note" id="more"></p>
</main>
<div id="big"><img id="bigimg"><div id="bigcap"></div></div>
<script>
const D=/*DATA*/;
const GN={1:"Landbird on Land",2:"Landbird on Water",3:"Waterbird on Land",4:"Waterbird on Water"};
const GT={1:2255,2:2255,3:642,4:642};
const S={g:null,sp:null,sc:null,minc:0.5};
const $=id=>document.getElementById(id);
const rows=D.rows;
$("tot").textContent=rows.length+" errors of 5794 test images";
function base(){return rows.filter(r=>(S.g==null||r.g==S.g)&&r.c>=S.minc)}
function cards(){
  $("cards").innerHTML=[1,2,3,4].map(g=>{const n=rows.filter(r=>r.g==g).length;
   return `<div class="card ${S.g==g?"sel":""}" data-g="${g}"><b>${n}</b>${GN[g]}<br><span>${(100*n/GT[g]).toFixed(1)}% of ${GT[g]} test images</span></div>`}).join("");
  document.querySelectorAll(".card").forEach(e=>e.onclick=()=>{S.g=S.g==e.dataset.g?null:+e.dataset.g;render()});
}
function hist(){
  const b=new Array(10).fill(0);base().forEach(r=>{b[Math.min(9,Math.floor((r.c-0.5)*20))]++});
  const m=Math.max(1,...b);
  $("hist").innerHTML=b.map((v,i)=>`<i style="height:${100*v/m}%" title="${(0.5+i*.05).toFixed(2)}-${(0.55+i*.05).toFixed(2)}: ${v}"></i>`).join("");
  $("hl").innerHTML=b.map((v,i)=>`<span>${(0.5+i*.05).toFixed(2)}<br>${v}</span>`).join("");
}
function bars(key,el,sk){
  const cnt={};base().forEach(r=>{if(S[sk=="sp"?"sc":"sp"]&&r[sk=="sp"?"sc":"sp"]!=S[sk=="sp"?"sc":"sp"])return;cnt[r[sk]]=(cnt[r[sk]]||0)+1});
  const T={};for(const k in D.totals[key]){const a=D.totals[key][k];T[k]=S.g?a[S.g-1]:a.reduce((x,y)=>x+y,0)}
  const list=Object.keys(T).filter(k=>T[k]>=5).map(k=>({k,n:cnt[k]||0,t:T[k],rate:(cnt[k]||0)/T[k]})).filter(x=>x.n>0).sort((a,b)=>b.rate-a.rate||b.n-a.n).slice(0,key=="species"?25:20);
  $(el).innerHTML=list.map(x=>`<div class="row ${S[sk]==x.k?"sel":""}" data-k="${x.k}"><div class="lab" title="${x.k}">${x.k}</div><div class="tr"><div class="fi" style="width:${Math.min(100,100*x.rate)}%"></div></div><div class="n">${x.n}/${x.t} (${(100*x.rate).toFixed(0)}%)</div></div>`).join("")||"<span class='note'>none</span>";
  document.querySelectorAll("#"+el+" .row").forEach(e=>e.onclick=()=>{S[sk]=S[sk]==e.dataset.k?null:e.dataset.k;render()});
}
function grid(){
  let r=base().filter(x=>(!S.sp||x.sp==S.sp)&&(!S.sc||x.sc==S.sc));
  const so=$("sort").value;
  r.sort(so=="chi"?(a,b)=>b.c-a.c:so=="clo"?(a,b)=>a.c-b.c:so=="sp"?(a,b)=>a.sp.localeCompare(b.sp):(a,b)=>a.sc.localeCompare(b.sc));
  const lim=+$("lim").value;
  $("chips").innerHTML=[S.g&&GN[S.g],S.sp,S.sc].filter(Boolean).map(x=>`<span class="chip">${x}</span>`).join("");
  $("grid").innerHTML=r.slice(0,lim).map(x=>`<div class="it"><img loading="lazy" src="images/${x.p}" data-cap="${x.sp} | ${x.sc} | true ${x.t}, pred ${x.d}, conf ${x.c}" alt=""><div><b>${x.sp.slice(4).replace(/_/g," ")}</b><br>${x.sc}<br>true ${x.t}, pred ${x.d}, ${x.c.toFixed(2)}</div></div>`).join("");
  $("more").textContent=`${r.length} matching, showing ${Math.min(lim,r.length)}`;
  document.querySelectorAll(".it img").forEach(i=>i.onclick=()=>{$("bigimg").src=i.src;$("bigcap").textContent=i.dataset.cap;$("big").style.display="flex"});
}
function render(){cards();hist();bars("scene","scenes","sc");bars("species","species","sp");grid()}
$("big").onclick=()=>{$("big").style.display="none"};
$("sort").onchange=$("lim").onchange=grid;
$("minc").oninput=e=>{S.minc=+e.target.value;$("mincv").textContent=S.minc.toFixed(2);render()};
$("clr").onclick=()=>{S.g=S.sp=S.sc=null;S.minc=0.5;$("minc").value=0.5;$("mincv").textContent="0.50";render()};
render();
</script></body></html>"""

if __name__ == "__main__":
    main()
