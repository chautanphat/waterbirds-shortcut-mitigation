"""Package saved real predictions into a self-contained Vietnamese team report."""
from pathlib import Path
import base64
import csv
import hashlib
import html
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/poter_failure'
OUTPUT=ROOT/'outputs/01a1ee6e-d18d-7d83-a20d-409d909f4897'
MODELS=['POTER','ERM_enhanced','DPE_legacy']
NAMES={'POTER':'POTER','ERM_enhanced':'ERM enhanced','DPE_legacy':'DPE legacy'}
LEVELS=['clean','mild','moderate','strong']
VN={'clean':'Không mờ','mild':'Nhẹ','moderate':'Vừa','strong':'Mạnh'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):
    with (SOURCE/name).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))
def num(row,key):return float(row[key]) if row[key] else None
def one(rows,**keys):
    found=[r for r in rows if all(r[k]==str(v) for k,v in keys.items())]
    if len(found)!=1:raise ValueError(('Ambiguous or missing metric',keys,len(found)))
    return found[0]


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    integrity=json.loads((SOURCE/'integrity.json').read_text(encoding='utf-8'))
    for name,expected in integrity['sha256'].items():
        if digest(SOURCE/name)!=expected:raise ValueError('Saved run changed: '+name)
    metrics=read('metrics.csv');original=read('original_test_metrics.csv');classes=read('class_metrics.csv')
    groups=read('group_metrics.csv');og=read('original_test_group_metrics.csv');ci=read('bootstrap_ci.csv')
    flips=read('flip_metrics.csv');species=read('species_metrics.csv')
    timings=json.loads((SOURCE/'timings.json').read_text())
    sections=[]
    def add(id,title,headers,rows,formats=None,notes=(),source=''):
        sections.append(dict(id=id,title=title,headers=headers,rows=rows,
            formats=formats or ['text']*len(headers),notes=list(notes),source=source))
    rows=[]
    for m in MODELS:
        r=one(original,model_name=m)
        correct=sum(int(g['correct']) for g in og if g['model_name']==m)
        rows.append([NAMES[m],int(r['n']),correct,num(r,'semantic_accuracy'),num(r,'wga'),
            num(r,'landbird_accuracy'),num(r,'waterbird_accuracy')])
    add('original','1. Toàn bộ test gốc: ảnh không chỉnh sửa',
        ['Model','Số ảnh','Số đúng','Accuracy','WGA','Landbird acc.','Waterbird acc.'],rows,
        ['text','int','int','pct','pct','pct','pct'],
        ['5.794 ảnh/model. Landbird: 4.510 ảnh; Waterbird: 1.284 ảnh. Nhãn official dùng cho bảng chính.',
         'Accuracy là tỷ lệ đúng trên số ảnh thực tế. WGA là accuracy thấp nhất trong bốn nhóm lớp × nền.'],
        'original_test_metrics.csv; original_test_group_metrics.csv')
    rows=[]
    for m in MODELS:
        for level in LEVELS:
            r=one(metrics,model_name=m,control='natural',severity=level)
            c=[one(classes,model_name=m,control='natural',severity=level,true_y=y) for y in (0,1)]
            rows.append([NAMES[m],VN[level],dict(clean=0,mild=1.5,moderate=4,strong=9)[level],
                num(r,'semantic_accuracy'),num(r,'wga'),num(r,'landbird_accuracy'),num(r,'waterbird_accuracy'),
                num(r,'predicted_waterbird_fraction'),num(c[0],'mean_p_waterbird'),num(c[1],'mean_p_waterbird')])
    add('gaussian','2. Gaussian foreground: so sánh theo mức mờ',
        ['Model','Mức mờ','Sigma (px)','Accuracy','WGA','Landbird acc.','Waterbird acc.',
         'Dự đoán Water','P(Water) | Land','P(Water) | Water'],rows,
        ['text','text','number']+['pct']*7,
        ['Mỗi hàng: 400 composites, từ 100 foreground (50 ảnh/lớp), 4 nền; 200 composites/lớp, 100 composites/nhóm.',
         'Không mờ ở đây là chim được tách mask rồi ghép nền, không phải 5.794 ảnh test gốc của bảng 1.',
         'Dự đoán Water = tỷ lệ pred_y=1. Hai cột cuối là xác suất Waterbird trung bình theo lớp chim thật.'],
        'metrics.csv; class_metrics.csv')
    rows=[]
    for m in MODELS:
        c=one(metrics,model_name=m,control='natural',severity='clean')
        s=one(metrics,model_name=m,control='natural',severity='strong')
        drops=[100*(num(c,k)-num(s,k)) for k in ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy']]
        rows.append([NAMES[m],*drops,drops[2]-drops[3]])
    add('drops','3. Mức giảm từ không mờ đến mờ mạnh',
        ['Model','Drop accuracy (pp)','Drop WGA (pp)','Drop Land (pp)','Drop Water (pp)','Lệch drop (pp)'],rows,
        ['text']+['number']*5,
        ['pp = điểm phần trăm. Drop = accuracy không mờ − accuracy mờ mạnh; số âm nghĩa là accuracy tăng.',
         'Lệch drop = Drop Landbird − Drop Waterbird. Số dương lớn cho thấy Landbird suy giảm mạnh hơn Waterbird.'],
        'Tính từ bảng 2; bootstrap_ci.csv')
    rows=[]
    for m in MODELS:
        for metric,label in [('wga_drop','Drop WGA'),('class_asymmetric_drop','Lệch drop hai lớp')]:
            r=one(ci,model_name=m,comparison='within_model',severity='strong',metric=metric)
            rows.append([NAMES[m],label,100*num(r,'estimate'),100*num(r,'ci95_low'),100*num(r,'ci95_high')])
    add('within_ci','4. Khoảng tin cậy trong từng model ở mức mờ mạnh',
        ['Model','Chỉ số','Ước lượng (pp)','CI 95% dưới (pp)','CI 95% trên (pp)'],rows,
        ['text','text','number','number','number'],
        ['Bootstrap 2.000 lần, resample foreground theo lớp (50 ảnh/lớp). Các nền và model cùng ảnh đi cùng nhau.',
         'Đơn vị độc lập trong bootstrap là source foreground, không phải từng composite.'], 'bootstrap_ci.csv')
    rows=[]
    for other in MODELS[1:]:
        for level in LEVELS[1:]:
            for metric,label in [('wga_drop','Drop WGA'),('class_asymmetric_drop','Lệch drop hai lớp')]:
                r=one(ci,comparison='POTER_minus_'+other,severity=level,metric=metric)
                lo,hi=100*num(r,'ci95_low'),100*num(r,'ci95_high')
                result='CI chứa 0' if lo<=0<=hi else 'POTER giảm lớn hơn' if lo>0 else 'POTER giảm ít hơn'
                rows.append(['POTER − '+NAMES[other],VN[level],label,100*num(r,'estimate'),lo,hi,result])
    add('between_ci','5. POTER khác baseline bao nhiêu?',
        ['So sánh','Mức mờ','Chỉ số','Chênh lệch (pp)','CI dưới (pp)','CI trên (pp)','Diễn giải'],rows,
        ['text','text','text','number','number','number','text'],
        ['Chênh lệch = Drop(POTER) − Drop(baseline). Dương: POTER giảm mạnh hơn; âm: POTER giảm ít hơn.',
         'CI chứa 0: chưa phân biệt rõ ở mẫu hiện tại. Các CI exploratory, chưa hiệu chỉnh nhiều so sánh.',
         'Kết quả có điều kiện trên checkpoint và 4 nền cố định, không ước lượng khác biệt giữa seed train.'], 'bootstrap_ci.csv')
    rows=[]
    for m in MODELS:
        for level in LEVELS:
            r=one(metrics,model_name=m,control='foreground_only',severity=level)
            rows.append([NAMES[m],VN[level],100,num(r,'semantic_accuracy'),num(r,'landbird_accuracy'),
                num(r,'waterbird_accuracy'),num(r,'predicted_waterbird_fraction'),num(r,'mean_p_waterbird')])
    add('foreground','6. Foreground-only: đặt chim trên nền xám trung tính',
        ['Model','Mức mờ','Số ảnh','Accuracy','Landbird acc.','Waterbird acc.','Dự đoán Water','P(Water) TB'],rows,
        ['text','text','int']+['pct']*5,
        ['Mỗi hàng có 100 ảnh. Neutral có nhãn chim để tính accuracy; không có WGA bốn nhóm Land/Water.'],
        'metrics.csv; class_metrics.csv')
    predictions=read('predictions.csv')
    rows=[]
    for m in MODELS:
        for bg in ['land','water']:
            unique={r['raw_input_sha256']:r for r in predictions
                if r['model_name']==m and r['control']=='background_only' and r['bg_type']==bg}
            rr=list(unique.values())
            assert len(rr)==2
            rows.append([NAMES[m],bg.title(),len(rr),sum(float(r['p_waterbird']) for r in rr)/len(rr),
                sum(int(r['pred_y'])==1 for r in rr)/len(rr),'Không áp dụng'])
    add('background','7. Background-only: nền không có chim',
        ['Model','Loại nền','Số nền duy nhất','P(Water) TB','Dự đoán Water','Accuracy / WGA'],rows,
        ['text','text','int','pct','pct','text'],
        ['Chỉ có 4 ảnh nền duy nhất (2 Land + 2 Water), được lặp theo 100 foreground trong file dự đoán.',
         'Đây là chẩn đoán xu hướng dự đoán ngoài miền. Không gán nhãn chim và không tính semantic accuracy/WGA.'],
        'predictions.csv: background_only, deduplicate raw_input_sha256 trong từng model')
    rows=[]
    for m in MODELS:
        for level in LEVELS[1:]:
            for y in (0,1):
                a=one(flips,model_name=m,severity=level,true_y=y,pair_type='clean_to_degraded')
                b=one(flips,model_name=m,severity=level,true_y=y,pair_type='land_background_to_water_background')
                rows.append([NAMES[m],VN[level],['Landbird','Waterbird'][y],num(a,'flip_rate'),
                    num(a,'land_to_water_rate'),num(a,'water_to_land_rate'),num(b,'flip_rate'),
                    100*num(b,'mean_delta_p_waterbird')])
    add('flips','8. Đổi dự đoán khi làm mờ và khi đổi nền',
        ['Model','Mức mờ','Lớp chim thật','Flip clean→blur','Pred Land→Water','Pred Water→Land','Flip đổi nền','ΔP(Water) (pp)'],
        rows,['text','text','text','pct','pct','pct','pct','number'],
        ['Clean→blur: so cùng foreground và cùng nền, 200 cặp/lớp. Hai cột hướng flip chia cho tất cả 200 cặp.',
         'Đổi nền: Land→Water giữ cùng foreground/severity, 100 cặp/lớp. ΔP = P(Water|nền Water) − P(Water|nền Land).',
         'Flip là đổi lớp dự đoán, không nhất thiết là chuyển từ đúng sang sai.'], 'flip_metrics.csv; paired_flips.csv')
    def group_rows(source):
        return [[NAMES[r['model_name']],VN.get(r['severity'],r['severity']),
            ['Landbird','Waterbird'][int(r['true_y'])],r['bg_type'].title(),int(r['n']),int(r['correct']),num(r,'accuracy')]
            for r in source if r['control'] in ['natural','original']]
    add('groups_original','9. Bốn nhóm trên test gốc',
        ['Model','Điều kiện','Lớp chim','Nền','Số ảnh','Số đúng','Accuracy'],group_rows(og),
        ['text']*4+['int','int','pct'],source='original_test_group_metrics.csv')
    add('groups_gaussian','10. Bốn nhóm trong thí nghiệm Gaussian',
        ['Model','Mức mờ','Lớp chim','Nền','Số ảnh','Số đúng','Accuracy'],
        group_rows([r for r in groups if r['control']=='natural']),
        ['text']*4+['int','int','pct'],source='group_metrics.csv')
    sensitivity=read('original_test_label_sensitivity.csv')
    add('labels','11. Phân tích nhạy cảm với lỗi nhãn species',
        ['Model','Chính sách nhãn','Số ảnh','Accuracy','WGA'],
        [[NAMES[r['model_name']],r['label_policy'],int(r['n']),num(r,'semantic_accuracy'),num(r,'wga')] for r in sensitivity],
        ['text','text','int','pct','pct'],
        ['Ba loài: Eastern Towhee, Western Meadowlark, Western Wood-Pewee bị official gán Waterbird do substring Tern.',
         'corrected: đánh giá lại ba loài thành Landbird và tính lại nhóm. exclude_known_errors: bỏ 90 ảnh test thuộc ba loài.',
         'Không sửa dataset/checkpoint và không chọn model theo các kết quả này. Bộ 100 foreground đã loại ba loài từ trước.'],
        'original_test_label_sensitivity.csv')
    add('reference100','12. Ảnh gốc của đúng 100 foreground trong replication',
        ['Model','Số ảnh','Accuracy','WGA','Landbird acc.','Waterbird acc.'],
        [[NAMES[m],100,*[num(one(metrics,model_name=m,control='original'),k) for k in
            ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy']]] for m in MODELS],
        ['text','int']+['pct']*4,
        ['100 ảnh này nằm trong 5.794 ảnh test gốc, không phải thêm 100 source độc lập.',
         'So với clean recomposition có cả thay nền và xử lý geometry/segmentation, nên delta không chỉ đo artifact mask.'],
        'metrics.csv: original')
    protocol=[
        ['Dữ liệu','Waterbirds official test, HF arubique/waterbirds; original CUB RGB + segmentation đã kiểm checksum.'],
        ['Chọn foreground','Seed 20261118. 50 ảnh/lớp, 10 loài/lớp, 5 ảnh/loài; loại 16 ảnh sơ bộ; không chọn theo lỗi model.'],
        ['Nền','2 Land + 2 Water, cùng bốn nền cho mọi foreground/model/severity. Chưa sàng lọc người/động vật trong nền.'],
        ['Can thiệp','Gaussian foreground sigma0/1.5/4/9 px. Mask alpha cố định. Không blur nền, không tạo ảnh bằng AI.'],
        ['Preprocess','RGB; resize256×256; center crop224×224; ImageNet normalization. Cả ba nhận cùng tensor SHA-256.'],
        ['POTER','ResNet50 + FC2; seed1; checkpoint validation-selected epoch258/300; LR1e-5, weight decay1.'],
        ['ERM enhanced','ResNet50 + FC2; seed0; enhanced/class-balanced ERM, augmentation, ImageNet V2, 400 epochs, LR0.001, WD0.'],
        ['DPE legacy','Cùng backbone ERM enhanced; 15 IsoMax prototype heads; equal mean probabilities; nhiệt độ inference1.'],
        ['DPE training pool','1.199 ID validation canonical, không có test ID. Ensemble cũ thiếu hash backbone lịch sử, cặp được khóa hiện tại.'],
        ['Seed model','Chỉ một checkpoint/model; seed POTER khác ERM/DPE. Chưa có xác nhận ba seed train độc lập khớp điều kiện.'],
        ['CI','2.000 bootstrap class-stratified theo source foreground, paired giữa model/severity; không resample composite độc lập.'],
        ['Thời gian thực đo',f"Full test gốc cả ba: {timings['full_original_test']['elapsed_seconds']:.1f}s; replication100 cả ba: {timings['replication100']['elapsed_seconds']:.1f}s."],
        ['Kiểm tra','21 tests passed; POTER Runner parity; DPE reference error1.19e-7; 5.794 source JPEG hash và 50 example crop hash được xác minh.'],
        ['Log POTER cũ','Full inference mới đúng5.330 ảnh, log Kaggle epoch258 đúng5.331; WGA khớp. Nguyên nhân chênh1 ảnh chưa xác định.'],
        ['Diễn giải','Các so sánh có điều kiện trên checkpoint/recipe và bốn nền cố định. Chưa xác lập khác biệt do thuật toán.'],
        ['Artifacts','Kiểm tra thủ công hai ảnh định trước; chưa duyệt mask của cả100. Có biên mask mềm, crop và vị trí ghép thiếu tự nhiên.'],
        ['Confidence','DPE score là log(mean head probability), không phải logit CNN. Xác suất giữa model chưa calibration chung.'],
        ['Cấu hình','poter_failure.yaml và experiment_config.yaml; source code/checkpoint/input hashes trong các file provenance.'],
    ]
    conclusions=[
        'Trên test gốc, DPE legacy đạt accuracy và WGA cao nhất trong ba checkpoint hiện có.',
        'Khi mờ mạnh, accuracy Landbird giảm ở cả10/10 loài Landbird với từng model. POTER và DPE giữ Waterbird cao.',
        'Strong: chênh lệch class-asymmetric drop POTER−DPE = +5,50 pp, CI95% [−6,50; +16,50]; chưa phân biệt rõ.',
        'Strong: POTER có lệch drop lớn hơn enhanced ERM +39,50 pp [28,00; 50,50], nhưng WGA drop không lớn hơn rõ ràng.',
        'Chưa có bằng chứng đây là lỗi riêng của POTER. Recipe/seed khác nhau và thiếu liên kết backbone DPE lịch sử giới hạn kết luận.',
    ]
    detail=[[NAMES[r['model_name']],['Landbird','Waterbird'][int(float(r['true_y']))],r['species_id'],VN[r['severity']],
        int(r['source_foregrounds']),int(r['images']),num(r,'accuracy'),num(r,'mean_p_waterbird')] for r in species]
    payload=dict(title='Waterbirds: so sánh POTER, ERM enhanced và DPE legacy',date='18/11/2026',
        conclusions=conclusions,sections=sections,protocol=protocol,
        species=dict(headers=['Model','Lớp chim','Species','Mức mờ','Số foreground','Số composite','Accuracy','P(Water) TB'],
                     rows=detail,formats=['text']*4+['int','int','pct','pct']))
    (OUTPUT/'comparison_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    def display(v,fmt):
        if v is None:return 'Không áp dụng'
        if fmt=='pct':return f'{v*100:.2f}%'
        if fmt=='number':return f'{v:.2f}'
        if fmt=='int':return f'{v:,}'
        return str(v)
    def table(headers,rows,formats):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td class="'+('num' if fmt!='text' else '')+'">'+html.escape(display(v,fmt))+'</td>' for v,fmt in zip(row,formats))+'</tr>' for row in rows)+'</tbody></table></div>'
    body='<header><p class="eyebrow">Kết quả inference thật · 18/11/2026</p><h1>'+payload['title']+'</h1><p>Checkpoint có sẵn. Không train lại và không chọn checkpoint theo test. Báo cáo gồm hai điều kiện đánh giá riêng.</p></header>'
    body+='<section class="takeaway"><h2>Nhận định từ kết quả</h2><ol>'+''.join('<li>'+html.escape(t)+'</li>' for t in conclusions)+'</ol></section>'
    body+='<section><h2>Đọc các bảng</h2><p><b>Test gốc:</b> 5.794 ảnh không chỉnh sửa. <b>Replication:</b> 100 foreground thuộc20 loài, tạo1.600 natural composites +400 neutral +400 background-only +100 ảnh gốc cho mỗi model. 100 ảnh gốc này nằm trong full test. Có24.882 dòng dự đoán được lưu, không phải24.882 source độc lập.</p><p><b>Accuracy/WGA:</b> cao hơn là tốt hơn. <b>Drop:</b> độ giảm từ clean composite đến mức blur, đơn vị điểm phần trăm (pp). <b>CI95%:</b> khoảng bất định theo foreground, không phản ánh seed model hay nền mới.</p></section>'
    for sec in sections:
        body+='<section><h2>'+html.escape(sec['title'])+'</h2>'+table(sec['headers'],sec['rows'],sec['formats'])
        body+=''.join('<p class="note">'+html.escape(t)+'</p>' for t in sec['notes'])
        body+='<p class="source">Nguồn: '+html.escape(sec['source'])+'</p></section>'
        if sec['id'] in ['gaussian','between_ci']:
            name='01_class_accuracy.png' if sec['id']=='gaussian' else '06_model_difference_ci.png'
            body+='<figure><img alt="'+html.escape(sec['title'])+'" src="data:image/png;base64,'+base64.b64encode((SOURCE/name).read_bytes()).decode()+'"></figure>'
    body+='<section><h2>13. Chi tiết từng loài</h2><p>Mỗi loài có5 foreground và20 composites mỗi mức mờ/model. Cùng source foreground xuất hiện ở bốn severity. Các hàng không độc lập giữa severity.</p>'+table(payload['species']['headers'],detail,payload['species']['formats'])+'<p class="source">Nguồn: species_metrics.csv</p></section>'
    body+='<section><h2>14. Phương pháp, kiểm tra và giới hạn</h2>'+table(['Mục','Nội dung'],protocol,['text','text'])+'</section>'
    css='''*{box-sizing:border-box}body{font-family:Arial,sans-serif;margin:0;color:#203047;background:#edf1f5;font-size:14px;line-height:1.55}main{max-width:1360px;margin:auto;padding:36px 40px;background:white}h1{font-size:30px;line-height:1.2;max-width:1000px}h2{font-size:20px;margin-top:0;color:#20334e}.eyebrow{color:#637082;font-size:12px}header{border-bottom:3px solid #304e76;padding-bottom:18px;margin-bottom:28px}section{margin:28px 0 38px}.takeaway{padding:22px;background:#f1f5fa;border-left:4px solid #345e87}li{margin:7px 0}table{border-collapse:collapse;width:100%;font-size:12px}th{text-align:left;background:#294563;color:white;line-height:1.35;padding:11px 10px;vertical-align:middle}td{padding:8px 10px;border-bottom:1px solid #dce3eb;vertical-align:top}tr:nth-child(even)td{background:#f5f7fa}.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.note{font-size:12px;color:#46576d;margin:7px 0}.source{font-size:11px;color:#6c7888}figure{margin:28px 0 40px}img{width:100%;height:auto}.scroll{overflow-x:auto}@media print{@page{size:A4 landscape;margin:12mm}body{background:white;font-size:11px}main{max-width:none;padding:0}h1{font-size:24px}h2{font-size:16px}table{font-size:9px}th,td{padding:5px 6px}thead{display:table-header-group}tr{break-inside:avoid}section{margin:18px 0}figure{break-inside:avoid}th{-webkit-print-color-adjust:exact;print-color-adjust:exact}.scroll{overflow:visible}}'''
    path=OUTPUT/'So_sanh_POTER_ERM_DPE_2026-11-18.html'
    path.write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>So sánh POTER ERM DPE</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
    provenance=dict(source_integrity_sha256=digest(SOURCE/'integrity.json'),source_files_verified=len(integrity['sha256']),
        builder_sha256=digest(Path(__file__)),new_inference=False,sections=len(sections),species_rows=len(detail),
        source_sha256={name:digest(SOURCE/name) for name in ['metrics.csv','class_metrics.csv','bootstrap_ci.csv',
            'original_test_metrics.csv','predictions.csv','species_metrics.csv']})
    (OUTPUT/'comparison_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__':main()
