const $=id=>document.getElementById(id),models=['POTER','ERM_enhanced','DPE_legacy'];
const names={POTER:'POTER',ERM_enhanced:'ERM enhanced',DPE_legacy:'DPE legacy'},level={clean:'Không mờ',mild:'Nhẹ',moderate:'Vừa',strong:'Mạnh',extreme:'Extreme (bổ sung)',original:'Ảnh gốc'};
const label=y=>y===null?'Không có nhãn chim':y===0?'Landbird':'Waterbird';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let rows=[],clean=new Map(),filtered=[],page=0,selected=null,info=null;
const cohort=()=>$('cohort').value;
const source=(r,view='model')=>'/api/asset?'+new URLSearchParams({cohort:cohort(),id:r.image_id,view});
async function guard(fn){$('error').textContent='';try{await fn()}catch(e){$('error').textContent=e.message}}
async function api(path){const response=await fetch(path);if(!response.ok)throw Error((await response.json()).error);return response.json()}
function wrong(r,m){return r.true_y!==null&&r.predictions[m].pred_y!==r.true_y}
function prediction(r,m){const p=r.predictions[m],style=r.true_y===null?'ood':wrong(r,m)?'wrong':'right';return `<span class="${style}">${label(p.pred_y)} · P(W) ${(100*p.p_waterbird).toFixed(2)}%</span>`}
function comparison(r){return `<table><thead><tr><th>Model</th><th>Dự đoán</th><th>P(Landbird)</th><th>P(Waterbird)</th><th>Kết quả</th></tr></thead><tbody>${models.map(m=>{const p=r.predictions[m];return `<tr><td>${names[m]}</td><td>${label(p.pred_y)}</td><td>${(100*p.p_landbird).toFixed(4)}%</td><td>${(100*p.p_waterbird).toFixed(4)}%</td><td class="${r.true_y===null?'ood':wrong(r,m)?'wrong':'right'}">${r.true_y===null?'OOD, không tính accuracy':wrong(r,m)?'Sai':'Đúng'}</td></tr>`}).join('')}</tbody></table>`}
async function load(){
  $('status').textContent='Đang đọc file dự đoán đã lưu…';
  rows=await api('/api/catalog?cohort='+cohort());
  const severityOptions=[...new Map(rows.map(r=>[r.severity,r.sigma])).entries()];
  severityOptions.sort((a,b)=>a[1]-b[1]||a[0].localeCompare(b[0]));
  $('severity').innerHTML='<option value="">Tất cả</option>'+severityOptions.map(([name,sigma])=>`<option value="${escape(name)}">${escape(level[name]||name)}${name==='original'?'':` (σ=${sigma})`}</option>`).join('');
  clean=new Map(rows.filter(r=>r.control==='natural'&&r.severity==='clean').map(r=>[r.source_image_id+'|'+r.background_id,r]));
  $('control').value=cohort()==='original'?'original':'natural';$('severity').value='';
  $('download').href='/api/download?cohort='+cohort();$('detail').hidden=true;filter();
  const bird=new URLSearchParams(location.search).get('bird');
  const first=filtered.find(r=>r.source_image_id===bird)||filtered[0];if(first)detail(first,false);
}
function filter(){
  page=0;const mode=$('mode').value,choice=$('model').value,active=choice==='any'?models:[choice],search=$('search').value.toLowerCase();
  filtered=rows.filter(r=>{
    if($('control').value!=='all'&&r.control!==$('control').value)return false;
    if($('class').value!==''&&String(r.source_y)!==$('class').value)return false;
    if($('severity').value&&r.severity!==$('severity').value)return false;
    if($('background').value&&r.bg_type!==$('background').value)return false;
    if(search&&!`${r.source_image_id} ${r.species}`.toLowerCase().includes(search))return false;
    if(mode==='failure')return active.some(m=>wrong(r,m));
    if(mode==='disagree')return new Set(models.map(m=>r.predictions[m].pred_y)).size>1;
    if(mode==='degraded'){
      const baseline=clean.get(r.source_image_id+'|'+r.background_id);
      return baseline&&r.severity!=='clean'&&active.some(m=>!wrong(baseline,m)&&wrong(r,m));
    }
    return true;
  });
  render();
}
function render(){
  $('status').textContent=`${filtered.length} ảnh khớp bộ lọc / ${rows.length} bản ghi ảnh, ba model/ảnh. Không chạy lại model.`;
  $('gallery').innerHTML=filtered.slice(page*40,(page+1)*40).map((r,i)=>`<tr data-index="${page*40+i}"><td>${escape(r.source_image_id)}</td><td>${escape(r.species)}</td><td>${label(r.true_y)}</td><td>${level[r.severity]||escape(r.severity)}</td><td>${escape(r.bg_type)}</td>${models.map(m=>'<td>'+prediction(r,m)+'</td>').join('')}</tr>`).join('');
  $('gallery').querySelectorAll('tr').forEach(tr=>tr.onclick=()=>detail(filtered[Number(tr.dataset.index)],true));
  $('page').textContent=`Trang ${page+1}/${Math.max(1,Math.ceil(filtered.length/40))}`;
  $('previous').disabled=page===0;$('next').disabled=(page+1)*40>=filtered.length;
}
function detail(r,scroll){
  selected=r;$('detail').hidden=false;
  $('case-title').textContent=`Source ${r.source_image_id} · ${r.species} · ${label(r.true_y)} · ${level[r.severity]||r.severity} (σ=${r.sigma}) / ${r.bg_type}`;
  $('case-csv').href='/api/download?'+new URLSearchParams({cohort:cohort(),bird:r.source_image_id});
  $('panes').innerHTML=[['Ảnh source gốc','source'],['Mask trên foreground','overlay'],['Đầu vào model chính xác 224 × 224','model']].map(([title,view])=>`<div><h3>${title}</h3><a href="${source(r,view)}" target="_blank" rel="noopener"><img src="${source(r,view)}" alt="${title}"></a></div>`).join('');
  $('comparison').innerHTML=comparison(r);
  const variants=rows.filter(v=>v.source_image_id===r.source_image_id&&v.control==='natural');
  $('variants').innerHTML=variants.length?variants.map(v=>`<div class="card"><h3>${level[v.severity]} (σ=${v.sigma}) / ${escape(v.bg_type)} / cặp ${v.pair_index+1}</h3><a href="${source(v)}" target="_blank" rel="noopener"><img loading="lazy" src="${source(v)}" alt="${escape(v.image_id)}"></a><table>${models.map(m=>`<tr><td>${names[m]}</td><td>${prediction(v,m)}</td></tr>`).join('')}</table></div>`).join(''):'<p class="muted">Bộ source gốc chỉ có ảnh không chỉnh sửa. Đổi sang bộ Gaussian để xem ảnh can thiệp và các mức mờ.</p>';
  $('provenance').textContent=JSON.stringify(r,null,2);
  if(scroll)$('detail').scrollIntoView({behavior:'smooth',block:'start'});
}
for(const id of ['control','mode','model','class','severity','background','search'])$(id).oninput=filter;
$('cohort').onchange=()=>guard(load);$('previous').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};
guard(async()=>{
  info=await api('/api/info');
  $('cohort').options[0].textContent=`${info.experiment}: ${info.cohorts.replication.source_foregrounds} foreground, ${info.cohorts.replication.species} loài`;
  $('cohort').options[1].textContent=`Ảnh source gốc: ${info.cohorts.original.source_foregrounds} ảnh`;
  await load();
});
