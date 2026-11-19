"""Build a Vietnamese report from saved runs only; no new model inference."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.counterfactual.common import read_csv,write_csv,write_json,sha256,pixel_hash

MAIN='20261118T123059Z_48f0272960'
DEMOS={'gaussian':'20261118T123339Z_06e2f340ac',
       'contrast':'20261118T125909Z_f754cb571a',
       'occlusion':'20261118T125922Z_7cc3576177'}
LEVELS=['clean','mild','moderate','strong']
GROUPS={'0':'Landbird / Land','1':'Landbird / Water','2':'Waterbird / Land','3':'Waterbird / Water'}


def yes(value):return value in (True,'True','true','1',1)
def pct(value):return 'Không xác định' if value is None else f"{(Decimal(str(value))*100).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)}%"
def val(value):return 'Không xác định' if value is None else f'{value:.4f}'
def ci(value):return 'Không xác định' if value is None else f'[{value[0]:.4f}; {value[1]:.4f}]'
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def read_run(root,run_id):
    path=root/run_id
    integrity=path/'integrity.json'
    if integrity.exists():
        record=json.loads(integrity.read_text(encoding='utf-8'))
        for name,digest in record['core_sha256'].items():
            if sha256(path/name)!=digest:raise ValueError(f'Run integrity mismatch: {run_id}/{name}')
    return dict(path=path,config=json.loads((path/'run_config.json').read_text(encoding='utf-8')),
        status=json.loads((path/'status.json').read_text(encoding='utf-8')),
        rows=read_csv(path/'predictions.csv'),pairs=read_csv(path/'paired_metrics.csv'),
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8')))


def figures(out,main,natural):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    x=np.arange(4)
    labels=['Clean\nσ=0','Mild\nσ=1.5','Moderate\nσ=4','Strong\nσ=9']
    fig=plt.figure(figsize=(12,8.2))
    grid=fig.add_gridspec(2,2)
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[1,:])
    a.plot(x,[100*s['accuracy'] for s in natural],'-o',color='#14765c',label='Overall accuracy')
    a.plot(x,[100*s['wga'] for s in natural],'--s',color='#a54a45',label='Worst-group accuracy')
    for i,s in enumerate(natural):a.annotate(pct(s['accuracy']),(i,100*s['accuracy']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    a.set_title('A  Accuracy on natural-background interventions',loc='left',fontweight='bold')
    a.legend(loc='lower left');a.set_ylabel('Accuracy (%)')
    for y,label,color in [(0,'Landbird (8 source birds)','#b05c36'),(1,'Waterbird (8 source birds)','#2878a4')]:
        values=[]
        for severity in LEVELS:
            subset=[r for r in main['rows'] if r['control']=='natural' and r['severity']==severity and int(r['true_y'])==y]
            values.append(100*sum(yes(r['is_correct']) for r in subset)/len(subset))
        b.plot(x,values,'-o',color=color,label=label)
    b.set_title('B  Accuracy by canonical bird class',loc='left',fontweight='bold')
    b.set_ylabel('Accuracy (%)');b.legend(loc='lower left')
    for axis in (a,b):
        axis.set_xticks(x,labels);axis.set_ylim(-4,111);axis.grid(axis='y',alpha=.2)
    means=np.array([s['D_s'] for s in natural]);bounds=np.array([s['D_s_ci95'] for s in natural])
    c.errorbar(x,means,yerr=np.vstack([means-bounds[:,0],bounds[:,1]-means]),fmt='o-',color='#176851',capsize=6)
    c.axhline(0,color='#777777',linestyle='--',linewidth=1)
    for i,value in enumerate(means):c.annotate(f'{value:.4f}',(i,value),xytext=(10,3),textcoords='offset points')
    c.set_title('C  Paired aligned − conflicting true-label margin (D_s)',loc='left',fontweight='bold')
    c.set_xticks(x,labels);c.set_ylabel('Logit margin difference');c.grid(axis='y',alpha=.2)
    fig.suptitle('POTER epoch 258 · controlled foreground blur · 16 source birds',fontweight='bold',fontsize=15)
    fig.text(.06,.025,'64 variants per blur level; 4 fixed backgrounds. Error bars: 95% bird-cluster bootstrap CI (1,000 resamples).\nTwo species, one checkpoint; retrospective exploratory test analysis. Controls excluded from accuracy.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.08,right=.97,top=.90,bottom=.14,hspace=.52,wspace=.30)
    fig.savefig(out/'gaussian_summary.png',dpi=220)
    fig.savefig(out/'gaussian_summary.svg')
    plt.close(fig)
    case=[r for r in main['rows'] if r['source_bird_id']=='179' and r['control']=='natural' and r['pair_index']=='0']
    assert len(case)==8
    fig,axes=plt.subplots(4,2,figsize=(8.2,11.5))
    for i,severity in enumerate(LEVELS):
        paired=[r for r in case if r['severity']==severity]
        assert len({r['foreground_layer_sha256'] for r in paired})==1
        for j,bg in enumerate(('land','water')):
            row=next(r for r in paired if r['bg_type']==bg)
            with Image.open(main['path']/row['model_input_path']) as im:
                im=im.convert('RGB')
                assert im.size==(224,224) and pixel_hash(im)==row['model_input_sha256']
                axes[i,j].imshow(np.asarray(im),interpolation='nearest')
            correct=yes(row['is_correct']);color='#176851' if correct else '#a52c36'
            axes[i,j].set_title(f"{severity} · {bg} · P(water)={float(row['p_waterbird']):.4f}\nPred={'waterbird' if int(row['pred_y']) else 'landbird'} · {'correct' if correct else 'error'}",fontsize=10,color=color)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            for spine in axes[i,j].spines.values():spine.set_visible(True);spine.set_color(color);spine.set_linewidth(1.5)
    fig.suptitle('Illustrative case #179 · true class: landbird · matched background pair 0',fontsize=12,fontweight='bold')
    fig.text(.07,.015,'Existing exact 224×224 model inputs; no new intervention or inference.\nLand: hf-bg-11699; Water: hf-bg-7032. Example selection does not change aggregate denominators.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96),h_pad=1.5)
    fig.savefig(out/'case_179_pair0.png',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=ROOT/'experiments/counterfactual_lab')
    p.add_argument('--output',type=Path,default=ROOT/'report/counterfactual_runs_2026-11-18')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    run=read_run(args.runs_root,MAIN)
    natural=[next(s for s in run['summary'] if s['control']=='natural' and s['severity']==level) for level in LEVELS]
    original=next(s for s in run['summary'] if s['control']=='original')
    totals=Counter(r['control'] for r in run['rows'])
    assert len(run['rows'])==400 and totals=={'natural':256,'original':16,'background_only':64,'foreground_only_on_neutral':64}
    aggregate=[]
    for summary in natural:
        records=[r for r in run['rows'] if r['control']=='natural' and r['severity']==summary['severity']]
        n=len(records);correct=sum(yes(r['is_correct']) for r in records)
        assert n==summary['valid_target_n'] and correct/n==summary['accuracy']
        pair_rows=[r for r in run['pairs'] if r['severity']==summary['severity']]
        assert len(pair_rows)==32 and sum(yes(r['paired_flip']) for r in pair_rows)==summary['paired_flips']
        by_class={y:[r for r in records if int(r['true_y'])==y] for y in (0,1)}
        aggregate.append(dict(severity=summary['severity'],sigma=run['config']['severities'][summary['severity']],n=n,correct=correct,
            failures=n-correct,accuracy=summary['accuracy'],wga=summary['wga'],paired_comparisons=len(pair_rows),paired_flips=summary['paired_flips'],
            landbird_accuracy=sum(yes(r['is_correct']) for r in by_class[0])/len(by_class[0]),
            waterbird_accuracy=sum(yes(r['is_correct']) for r in by_class[1])/len(by_class[1]),
            D_s=summary['D_s'],amplification=summary['amplification'],D_s_ci95=json.dumps(summary['D_s_ci95']),
            amplification_ci95=json.dumps(summary['amplification_ci95'])))
    assert sum(r['failures'] for r in aggregate)==80
    inventory=[]
    for path in sorted(args.runs_root.iterdir()):
        if (path/'run_config.json').exists():
            cfg=json.loads((path/'run_config.json').read_text());st=json.loads((path/'status.json').read_text())
            inventory.append(dict(run_id=path.name,operator=cfg.get('operator','gaussian'),source_birds=len(cfg['foreground_ids']),
                rows=st.get('rows'),status=st['status'],role='main_cohort' if path.name==MAIN else 'representative_smoke' if path.name in DEMOS.values() else 'repeated_QA'))
    write_csv(out/'run_inventory.csv',inventory);write_csv(out/'gaussian_metrics.csv',aggregate)
    group_rows=[dict(severity=s['severity'],group_id=g,group_name=GROUPS[g],**values) for s in natural for g,values in s['group_stats'].items()]
    write_csv(out/'group_metrics.csv',group_rows)
    bg=defaultdict(list)
    for r in run['rows']:
        if r['control']=='background_only':bg[(r['background_id'],r['bg_type'])].append(float(r['p_waterbird']))
    bg_rows=[dict(background_id=key[0],bg_type=key[1],repeat_rows=len(values),mean_p_waterbird=sum(values)/len(values)) for key,values in bg.items()]
    write_csv(out/'background_only_diagnostics.csv',bg_rows)
    originals=[r for r in run['rows'] if r['control']=='original']
    species=Counter(r['species'] for r in originals)
    flip_birds={r['source_bird_id'] for r in run['pairs'] if yes(r['paired_flip'])}
    demo_rows=[]
    for operator,run_id in DEMOS.items():
        demo=read_run(args.runs_root,run_id)
        rows=[r for r in demo['rows'] if r['control']=='natural']
        demo_rows.append([operator,run_id,len(demo['rows']),len(rows),sum(yes(r['is_correct']) for r in rows),sum(yes(p['paired_flip']) for p in demo['pairs'])])
    figures(out,run,natural)
    relative=lambda path:Path(os.path.relpath(path,out)).as_posix()
    main_link=relative(run['path'])
    parity=json.loads((ROOT/'experiments/counterfactual_verification/live_runner_parity.json').read_text())
    model=run['config']['models'][0]
    accuracy_table=mdtable(['Mức blur','σ','Đúng / tổng','Accuracy','WGA','Lỗi','Flip / cặp'],
        [[r['severity'],r['sigma'],f"{r['correct']}/{r['n']}",pct(r['accuracy']),pct(r['wga']),r['failures'],f"{r['paired_flips']}/{r['paired_comparisons']}"] for r in aggregate])
    class_table=mdtable(['Mức','Landbird — 32 biến thể','Waterbird — 32 biến thể'],[[r['severity'],pct(r['landbird_accuracy']),pct(r['waterbird_accuracy'])] for r in aggregate])
    group_table=mdtable(['Nhóm — 16 biến thể / mức']+[x.title() for x in LEVELS],
        [[GROUPS[g]]+[f"{s['group_stats'][g]['correct']}/16 ({pct(s['group_stats'][g]['accuracy'])})" for s in natural] for g in GROUPS])
    margin_table=mdtable(['Mức','D_s','CI 95% của D_s','Amplification','CI 95% của Amplification'],
        [[s['severity'],val(s['D_s']),ci(s['D_s_ci95']),val(s['amplification']),ci(s['amplification_ci95'])] for s in natural])
    inventory_table=mdtable(['Run ID — UTC','Operator','Chim nguồn','Dòng dự đoán','Vai trò'],
        [[r['run_id'],r['operator'],r['source_birds'],r['rows'],r['role']] for r in inventory])
    smoke_table=mdtable(['Operator','Run ID','Tổng dòng','Biến thể tự nhiên','Đúng','Paired flips'],demo_rows)
    bg_table=mdtable(['Background ID','Loại nền','Số dòng lặp','P(waterbird) trung bình'],
        [[r['background_id'],r['bg_type'],r['repeat_rows'],f"{r['mean_p_waterbird']:.4f}"] for r in bg_rows])
    text=f'''# Báo cáo các run Counterfactual Lab và Failure Case Explorer

**Ngày báo cáo:** 18/11/2026 · **Phạm vi:** tổng hợp các run đã lưu; không chạy thêm suy luận hoặc huấn luyện để lập báo cáo này.

## 1. Kết quả chính

Pipeline đã chạy với checkpoint POTER thật và hoàn tất cả quy trình tạo ảnh có kiểm soát, suy luận, lưu run, mở lại trong Explorer và xuất kết quả.

Run chính gồm **16 chim nguồn, 400 dòng dự đoán và 340 ảnh đầu vào duy nhất theo hash**. Trong đó có 256 biến thể chim trên nền tự nhiên, 64 dòng nền không có chim, 64 dòng chim trên nền trung tính và 16 ảnh gốc. Nền không có chim chỉ có bốn ảnh duy nhất, lặp lại theo 16 chim nguồn; không được coi là 64 mẫu độc lập.

Trên 256 biến thể nền tự nhiên, mô hình có **80 lỗi phân loại**. Có **10 lần đổi nhãn dự đoán giữa hai nền trong 128 so sánh cặp**, xuất hiện ở **{len(flip_birds)} chim nguồn**; đây không phải 10 chim độc lập. Ảnh gốc có một lỗi riêng, ở chim #192. Hai loại lỗi này không gộp chung mẫu số accuracy.

Khi làm mờ foreground, accuracy giảm từ **96,88% ở clean** xuống **53,13% ở moderate** và **54,69% ở strong**. Sự giảm không đơn điệu ở mọi mức. Các lỗi trên ảnh can thiệp đều là **Landbird bị dự đoán thành Waterbird** trong cohort này.

**Diễn giải nghiên cứu:** kết quả cho thấy độ nhạy với suy giảm chi tiết chim và một số can thiệp nền. Chỉ số margin đã định nghĩa chưa cho thấy tăng ưu tiên nền đồng nhãn: D_s và Amplification trung bình không tăng dương khi blur mạnh hơn. Không kết luận từ các run này rằng POTER nói chung phụ thuộc nền nhiều hơn khi mất tín hiệu foreground.

## 2. Thiết lập và dữ liệu

- Mô hình: POTER / ResNet-50, checkpoint epoch **258**, chọn theo validation trước thí nghiệm; không chọn hoặc điều chỉnh mô hình bằng nhãn test.
- SHA-256 checkpoint: `{model['checkpoint_sha256']}`.
- Nhãn chuẩn: `y=0` Landbird, `y=1` Waterbird; `place=0` Land, `place=1` Water.
- Tiền xử lý: resize vuông 256×256 → center crop 224×224 → tensor RGB → ImageNet normalization. Lưu đúng ảnh RGB 224×224 trước normalization.
- Dữ liệu đã kiểm tra: 17.382 ảnh Hugging Face; 5.794 bộ scene/foreground/background ghép được với metadata. Ngân hàng có 2.897 nền Land và 2.897 nền Water.
- Foreground: 5.794 cutout từ **RGB và mask CUB gốc đã xác minh**, không dùng ảnh sinh bằng AI. Tất cả ảnh HF cung cấp thuộc official test, dù tên loader HF là `train`.
- Hai tập nền trùng pixel chính xác được ghi nhận và loại khỏi lấy mẫu tự động; 63 tập va chạm average-hash là ứng viên cần kiểm tra, không phải xác nhận trùng ngữ nghĩa.

Cohort chính được ấn định bằng **tám ID test đầu tiên theo số cho mỗi lớp**, trước khi chạy inference; seed lấy nền là 2026. Có hai nền Land và hai nền Water, tái sử dụng cho mọi chim và mức blur. Cohort thực tế chỉ gồm **8 Groove-billed Ani (Landbird)** và **8 Black-footed Albatross (Waterbird)**. Vì vậy lớp và loài bị gắn với nhau trong mẫu nhỏ này.

Các mức Gaussian là clean σ=0, mild σ=1,5, moderate σ=4 và strong σ=9, tính theo pixel ở model view. Alpha, vị trí, kích thước và opacity giữ cố định giữa hai nền tại cùng mức. Blur chỉ tác động lên foreground bằng normalized convolution; các kiểm tra xác nhận RGBA layer giống nhau, foreground opaque bất biến và vùng ngoài mask bằng nền đã chọn.

## 3. Kết quả run chính

### 3.1. Accuracy và WGA trên nền tự nhiên

{accuracy_table}

Mỗi mức có 64 biến thể được tạo từ **16 ID ảnh chim nguồn**. Mỗi ảnh nguồn xuất hiện trên bốn nền. WGA được tính trên bốn nhóm `(nhãn chim, loại nền)`, mỗi nhóm có 16 biến thể; WGA bằng 0 ở moderate/strong vì nhóm Landbird/Land có 0/16 đúng, không phải vì thiếu nhóm.

Ảnh gốc được đánh giá riêng: **15/16 đúng = {pct(original['accuracy'])}**, WGA **{pct(original['wga'])}**. Mẫu số từng nhóm ảnh gốc là 3, 5, 5 và 3, khác với biến thể can thiệp. Không coi chênh lệch giữa accuracy ảnh gốc và clean recomposite là cải thiện hiệu năng trên cùng một benchmark.

### 3.2. Phân rã theo lớp và nhóm

{class_table}

{group_table}

Tổng 80 lỗi can thiệp phân bố trên cả tám Landbird. Tám Waterbird đều được dự đoán đúng trong các biến thể đã chạy. Điều này phù hợp với xu hướng dự đoán Waterbird khi chi tiết của các Landbird trong cohort bị làm yếu; dữ liệu hiện tại chưa tách được ảnh hưởng của lớp, loài và đặc điểm hình ảnh.

![Accuracy, WGA và paired margin của run chính](gaussian_summary.png)

### 3.3. Paired margin và mức khuếch đại

Định nghĩa `m_y = logit(y thật) − logit(lớp còn lại)`; `D_s = mean[m_y(nền đồng nhãn) − m_y(nền trái nhãn)]`; `Amplification(s) = D_s − D_clean`.

{margin_table}

CI được bootstrap **1.000 lần theo chim nguồn**, sau khi lấy trung bình các cặp của từng chim. Các CI này điều kiện trên bốn nền cố định và một checkpoint; không phản ánh bất định do chọn ngân hàng nền mới hoặc seed huấn luyện mới.

D_s âm có nghĩa margin trung bình nghiêng về nền trái nhãn trong những cặp đã chọn. Amplification tại mild âm; CI của Amplification tại moderate và strong chứa 0. Các kết quả này không hỗ trợ phát biểu “blur làm tăng ưu tiên nền đồng nhãn” trên cohort hiện tại. Paired flips là dấu hiệu nhạy với can thiệp nền, không tự xác lập nguyên nhân shortcut.

## 4. Các run một chim và controls

{smoke_table}

Cả ba run đại diện dùng cùng Waterbird #1, với tám biến thể nền tự nhiên mỗi run. Tất cả tám biến thể đều đúng và không có paired flip. Đây là smoke test trên một cá thể đã xuất hiện trong cohort chính; không cộng như dữ liệu độc lập. WGA không xác định vì thiếu hai nhóm Landbird; CI theo chim nguồn cũng không xác định với một chim.

Contrast dùng hệ số 1 / 0,75 / 0,4 / 0,1 quanh trung bình màu foreground có trọng số alpha. Occlusion che 0 / 10 / 30 / 50% pixel trong alpha support bằng RGB 127, với tâm và thứ tự che cố định theo seed. Hai operator này nằm trong run và chart riêng, không trộn với Gaussian. Ảnh clean của cả ba operator đã được đối chiếu và giống nhau theo byte ở cùng geometry và nền.

### Controls nền không có chim của run chính

{bg_table}

Đây là bốn ảnh nền duy nhất, mỗi ảnh được chạy lặp 16 lần theo cấu trúc factorial. Cả bốn có P(waterbird)>0,5, kể cả nền Land. Bộ phân loại đóng vẫn phải trả phân phối trên hai lớp chim dù không có chim. Các con số này là **chẩn đoán OOD**, không có nhãn chim thật để tính accuracy hoặc WGA. Chim trên nền xám cũng được lưu như control riêng và bị loại khỏi các mẫu số accuracy.

## 5. Minh họa một failure case

Chim **#179**, nhãn thật Landbird, cặp nền 0: Land `hf-bg-11699`, Water `hf-bg-7032`. Ở clean, P(waterbird) thay đổi từ **0,4598** sang **0,5178**, làm đổi nhãn dự đoán. Ở mild, cả hai nền đều bị dự đoán thành Waterbird. Các mức khác được hiển thị đầy đủ dưới đây.

![Case #179: ảnh model thực sự nhìn thấy tại bốn mức blur và hai nền](case_179_pair0.png)

Đây là minh họa hậu nghiệm từ kết quả đã lưu; không được dùng thay thống kê của toàn bộ 256 biến thể. Flip clean này ở gần ngưỡng 0,5, nên không nên diễn giải như một thay đổi xác suất lớn. Hình dùng các PNG 224×224 đã có, không tạo can thiệp mới hoặc chạy lại mô hình.

## 6. Kiểm tra tính đúng và phần mềm

- Đối chiếu với POTER Runner đang chạy: **pixel model view giống hoàn toàn**; sai lệch xác suất tối đa trên cùng CPU là **{parity['same_device_max_probability_error']:.3e}**. CPU so với CUDA lệch tối đa **{parity['cpu_vs_cuda_max_probability_error']:.3e}**, được ghi riêng như sai khác số học giữa thiết bị.
- Đối chiếu độc lập với loader/transform Runner trước refactor: cùng GPU batch cho sai lệch logits bằng 0. Repeated inference cùng batch cho sai lệch xác suất bằng 0; batch size 1/2 nằm trong tolerance đã lưu.
- Bộ test áp dụng cho Windows: **241 passed, 9 skipped, 5 deselected, 27 subtests passed**. Bộ acceptance của app: **15 passed**, gồm inference thật và ba operator. Tám kiểm tra Mixer và kiểm tra serialization của controls UI đều pass.
- Năm test bị loại là các test orchestration/notebook cũ phụ thuộc Linux hoặc cách ghi đường dẫn Unix. Danh sách và lệnh nằm trong [báo cáo kỹ thuật](../../docs/counterfactual_validation.md).
- Đã mở hai app, tạo run, đọc lại sau khi lưu, lọc failure, xem exact crop và xuất PNG/PDF/CSV/JSON. Một case packet #190 có 25 row ID và xác suất khớp giữa CSV, JSON và metadata PNG.
- Setup offline giữ nguyên các hash mẫu; snapshot vật lý có danh mục SHA-256 để tránh tải lại dữ liệu không đổi. Baseline POTER không bị sửa và không có huấn luyện mới.

## 7. Danh mục toàn bộ run đã lưu

{inventory_table}

Tổng có **{len(inventory)} run hoàn tất và {sum(r['rows'] for r in inventory)} dòng dự đoán được lưu**. Những run `repeated_QA` phục vụ kiểm tra browser/API hoặc lặp lại pipeline; không gộp chúng vào ước lượng hiệu năng độc lập. Thống kê ở mục 3 chỉ dùng run chính 400 dòng. Run ID dùng thời gian UTC; giờ Việt Nam bằng UTC+7.

## 8. Giới hạn và phạm vi kết luận

1. Đây là `retrospective_exploratory`: dữ liệu test đã được dùng để quan sát, không phải xác nhận độc lập trên test mới.
2. Chỉ 16 chim, hai loài, bốn nền cố định và một checkpoint. Các biến thể lặp theo chim; không phải 256 chim độc lập hoặc đánh giá toàn bộ 5.794 chim.
3. Nền chưa được lọc tự động về người/động vật; một số ảnh nền có người. Các tín hiệu bối cảnh cùng thay đổi khi đổi ảnh nền, không chỉ nhãn Land/Water trừu tượng.
4. Không có so sánh DPE thực nghiệm vì thiếu cặp heads/backbone tương thích có hash ràng buộc. Adapter đã có nhưng không tạo dự đoán thay thế.
5. Kết quả hỗ trợ việc xác nhận pipeline, tìm failure và đặt giả thuyết về tín hiệu foreground. Chưa đủ để khẳng định hiệu năng toàn Waterbirds, ưu thế của POTER/DPE, hoặc quan hệ nhân quả tổng quát về shortcut amplification.

## 9. Tệp bằng chứng và tái tạo báo cáo

- [Predictions run chính]({main_link}/predictions.csv), [manifest]({main_link}/manifest.csv), [paired metrics]({main_link}/paired_metrics.csv), [summary]({main_link}/summary.json), [cấu hình]({main_link}/run_config.json).
- [CSV chỉ số Gaussian](gaussian_metrics.csv), [CSV chỉ số nhóm](group_metrics.csv), [CSV controls nền](background_only_diagnostics.csv), [danh mục run](run_inventory.csv).
- [Nguồn dữ liệu và kiểm tra tải](../../DATASET_PROVENANCE.md), [chi tiết triển khai/kiểm thử](../../docs/counterfactual_validation.md).

Tạo lại đúng báo cáo từ các run đã lưu, không chạy mô hình:

```powershell
.\\.venv\\Scripts\\python.exe scripts/report_counterfactual_runs.py
```

Các run gốc được giữ nguyên. Báo cáo và hình ở đây là sản phẩm tổng hợp từ CSV/JSON/PNG hiện có.
'''
    (out/'report.vi.md').write_text(text,encoding='utf-8')
    write_json(out/'report_provenance.json',dict(source_run=MAIN,representative_demos=DEMOS,
        all_runs=inventory,script_sha256=sha256(__file__),no_new_inference=True,
        source_files_sha256={name:sha256(run['path']/name) for name in ['predictions.csv','paired_metrics.csv','summary.json','run_config.json']},
        source_species=dict(species),unique_raw_inputs=len({r['raw_input_sha256'] for r in run['rows']}),
        natural_failures=80,paired_flips=10,flip_source_birds=sorted(flip_birds,key=int)))
    print(json.dumps(dict(report=str(out/'report.vi.md'),figures=['gaussian_summary.png','case_179_pair0.png'],
        main_rows=len(run['rows']),unique_inputs=340,saved_runs=len(inventory),saved_prediction_rows=sum(r['rows'] for r in inventory),no_new_inference=True),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
