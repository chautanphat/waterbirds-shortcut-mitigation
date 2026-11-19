"""Validate frozen three-method outputs, compute paired CIs and six report plots."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.counterfactual.common import read_csv,sha256,write_json
from src.counterfactual.failure_statistics import (validate_predictions,metric_tables,paired_flips,bootstrap_comparison)
from src.counterfactual.investigation import MODELS,LEVELS


def table(data,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
        ['| '+' | '.join('—' if pd.isna(r[c]) else f'{r[c]:.2f}' if isinstance(r[c],(float,np.floating)) else str(r[c])
                       for c in columns)+' |' for _,r in data.iterrows()])


def plots(out,summary,classes,flips,ci):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    colors={'POTER':'#b64b39','ERM_enhanced':'#2878a4','DPE_legacy':'#437e4c'}
    x=np.arange(4)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained',sharey=True)
    for ax,model in zip(axes,MODELS):
        for y,label,color in [(0,'Landbird','#b86e31'),(1,'Waterbird','#247fa3')]:
            s=classes[(classes.model_name==model)&(classes.control=='natural')&(classes.true_y==y)].set_index('severity').loc[list(LEVELS)]
            ax.plot(x,100*s.accuracy,'-o',label=label,color=color)
        ax.set(title=model,xticks=x,xticklabels=LEVELS,ylim=(-3,103),ylabel='Accuracy (%)');ax.legend()
    fig.suptitle('Foreground Gaussian blur: class accuracy | 50 foregrounds/class, 10 species/class')
    save(fig,'01_class_accuracy')
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for model in MODELS:
        s=summary[(summary.model_name==model)&(summary.control=='natural')].set_index('severity').loc[list(LEVELS)]
        ax.plot(x,100*s.wga,'-o',label=model,color=colors[model])
    ax.set(xticks=x,xticklabels=LEVELS,ylim=(-3,103),ylabel='Worst-group accuracy (%)',
        title='Four equally sized class × background groups (100 images/group/severity)');ax.legend()
    save(fig,'02_wga')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained',sharey=True)
    for y,ax in enumerate(axes):
        for model in MODELS:
            s=classes[(classes.model_name==model)&(classes.control=='natural')&(classes.true_y==y)].set_index('severity').loc[list(LEVELS)]
            ax.plot(x,s.mean_p_waterbird,'-o',label=model,color=colors[model])
        ax.axhline(.5,color='gray',lw=.7,ls=':');ax.set(xticks=x,xticklabels=LEVELS,ylim=(0,1),
            ylabel='Mean P(Waterbird)',title='True '+['Landbird','Waterbird'][y]);ax.legend()
    save(fig,'03_waterbird_probability')
    fig,axes=plt.subplots(3,3,figsize=(10,8),layout='constrained')
    for mi,model in enumerate(MODELS):
        for si,severity in enumerate(LEVELS[1:]):
            ax=axes[mi,si]
            f=flips[(flips.model_name==model)&(flips.severity==severity)&(flips.pair_type=='clean_to_degraded')]
            counts=np.array([[((f.from_pred==i)&(f.to_pred==j)).sum() for j in (0,1)] for i in (0,1)])
            ax.imshow(counts,cmap='Blues',vmin=0,vmax=400)
            for i in (0,1):
                for j in (0,1):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>220 else 'black')
            ax.set(title=model+' / '+severity,xticks=[0,1],xticklabels=['Land','Water'],
                yticks=[0,1],yticklabels=['Land','Water'],xlabel='Degraded prediction',ylabel='Clean prediction')
    fig.suptitle('Paired prediction transitions (400 composites/cell; 100 source clusters)')
    save(fig,'04_flip_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for model in MODELS:
        for y,style in [(0,'-o'),(1,'--s')]:
            s=classes[(classes.model_name==model)&(classes.control=='foreground_only')&(classes.true_y==y)].set_index('severity').loc[list(LEVELS)]
            axes[0].plot(x,100*s.accuracy,style,color=colors[model],label=model+' / '+['Land','Water'][y])
        bg=summary[(summary.model_name==model)&(summary.control=='background_only')].iloc[0]
        axes[1].bar(MODELS.index(model),bg.mean_p_waterbird,color=colors[model],label=model)
    axes[0].set(xticks=x,xticklabels=LEVELS,ylim=(-3,103),ylabel='Semantic accuracy (%)',title='Foreground on neutral gray')
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=x[:3],xticklabels=['POTER','ERM enhanced','DPE legacy'],ylim=(0,1),ylabel='Mean P(Waterbird)',
        title='Background-only: 4 unique backgrounds, no bird')
    axes[1].axhline(.5,color='gray',ls=':',lw=.7)
    fig.suptitle('Isolation controls | repeated background-only rows are not independent observations')
    save(fig,'05_isolation_controls')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['wga_drop','class_asymmetric_drop'],['Difference in WGA drop','Difference in class-asymmetric drop']):
        for oi,other in enumerate(MODELS[1:]):
            d=ci[(ci.comparison=='POTER_minus_'+other)&(ci.metric==metric)].set_index('severity').loc[list(LEVELS)]
            ax.errorbar(x+(oi-.5)*.08,100*d.estimate,
                yerr=100*np.array([d.estimate-d.ci95_low,d.ci95_high-d.estimate]),fmt='o-',capsize=3,
                label='POTER − '+other,color=colors[other])
        ax.axhline(0,color='black',lw=.8);ax.set(xticks=x,xticklabels=LEVELS,ylabel='Difference (percentage points)',title=title);ax.legend(fontsize=8)
    fig.suptitle('Paired class-stratified foreground bootstrap | 95% percentile CI, 2,000 resamples')
    save(fig,'06_model_difference_ci')


def inspection_figure(out):
    """Preselected examples show boundaries and identical contextual interventions."""
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(2,7,figsize=(16,5.7),layout='constrained')
    for axes_row,folder in zip(axes,sorted((out/'examples').iterdir())):
        bird=folder.name.removeprefix('hf-fg-')
        names=[(bird+'_original.png','Original scene'),('mask.png','CUB mask'),
            ('mask_overlay.png','Mask overlay'),(bird+'_clean_hf-bg-11699.png','Clean / Land'),
            (bird+'_clean_hf-bg-7032.png','Clean / Water'),
            (bird+'_strong_hf-bg-11699.png','Strong / Land'),(bird+'_strong_hf-bg-7032.png','Strong / Water')]
        for ax,(name,title) in zip(axes_row,names):
            with Image.open(folder/name) as im:ax.imshow(im,cmap='gray' if name=='mask.png' else None)
            ax.set_title(title,fontsize=9);ax.axis('off')
        axes_row[0].set_title('Source '+bird+'\nOriginal scene',fontsize=9)
    fig.suptitle('Examples fixed before inference | source geometry, segmentation and background interventions')
    fig.savefig(out/'07_composition_inspection.png',dpi=160,bbox_inches='tight');plt.close(fig)


def inspect_saved_inputs(out,frame,config):
    """Verify exported example pixels against real model-view hashes, audit size."""
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from src.datasets.waterbirds import waterbirds_transform
    from src.counterfactual.common import pixel_hash
    from src.counterfactual.compositor import foreground_layer
    from src.counterfactual.data import ForegroundRepository
    preview=transforms.Compose(waterbirds_transform().transforms[:2])
    repo=ForegroundRepository(ROOT/config['data_root'])
    cohort=pd.read_csv(out/'cohort.csv')
    mask_stats=[]
    for row in cohort.itertuples():
        rgba,meta=repo.get(row.foreground_id)
        layer=foreground_layer(rgba,0)
        alpha=np.asarray(layer.getchannel('A'),dtype=float)/255
        cropped=alpha[16:240,16:240]
        mask_stats.append(dict(source_image_id=row.bird_source_id,species_id=row.species,true_y=row.y,
            mask_sha256=meta['mask_sha256'],mask_support_model_pixels=int((cropped>0).sum()),
            model_crop_alpha_area_fraction=float(cropped.mean()),
            alpha_mass_retained_by_center_crop=float(cropped.sum()/alpha.sum())))
    pd.DataFrame(mask_stats).to_csv(out/'mask_quality_audit.csv',index=False)
    examples=sorted((out/'examples').iterdir())
    fig,axes=plt.subplots(2,4,figsize=(10,5.5),layout='constrained')
    checks=0
    for axes_row,folder in zip(axes,examples):
        bird=folder.name.removeprefix('hf-fg-')
        records=frame[(frame.model_name=='POTER')&(frame.foreground_id==folder.name)]
        for row in records.itertuples():
            with Image.open(folder/(row.image_id+'.png')) as image:
                exact=preview(image.convert('RGB'))
            if pixel_hash(exact)!=row.model_input_sha256:
                raise ValueError('Saved example does not match real model-view input')
            checks+=1
        for ax,(severity,bg) in zip(axes_row,[('clean','11699'),('clean','7032'),('strong','11699'),('strong','7032')]):
            with Image.open(folder/f'{bird}_{severity}_hf-bg-{bg}.png') as image:
                ax.imshow(preview(image.convert('RGB')))
            ax.set_title(f"{bird}: {severity} / {'Land' if bg=='11699' else 'Water'}",fontsize=9);ax.axis('off')
    fig.suptitle('Exact 224 × 224 model views | PNG example hashes verified against inference')
    fig.savefig(out/'08_exact_model_inputs.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return dict(example_model_view_hashes_verified=checks,masks_verified=len(mask_stats))


def report(out,summary,classes,ci,original,original_sensitivity,checks,config,species):
    import json
    audit=json.loads((out/'checkpoint_audit.json').read_text(encoding='utf-8'))
    timings=json.loads((out/'timings.json').read_text())
    natural=summary[summary.control=='natural'].copy()
    natural['land_drop_pp']=natural.apply(lambda r:100*(natural[(natural.model_name==r.model_name)&(natural.severity=='clean')].iloc[0].landbird_accuracy-r.landbird_accuracy),axis=1)
    natural['water_drop_pp']=natural.apply(lambda r:100*(natural[(natural.model_name==r.model_name)&(natural.severity=='clean')].iloc[0].waterbird_accuracy-r.waterbird_accuracy),axis=1)
    for c in ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy','predicted_waterbird_fraction']:
        natural[c]*=100
    orig=original.copy()
    for c in ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy']:orig[c]*=100
    cohort=pd.read_csv(out/'cohort.csv')
    affected=original_sensitivity.copy()
    for c in ['semantic_accuracy','wga']:affected[c]*=100
    delta=ci[(ci.comparison!='within_model') & (ci.metric.isin(['wga_drop','class_asymmetric_drop'])) & (ci.severity!='clean')].copy()
    for c in ['estimate','ci95_low','ci95_high']:delta[c]*=100
    text=('# Báo cáo so sánh POTER / ERM enhanced / DPE legacy\n\n'
        '**Đã chạy inference thật với weight lưu sẵn, không train, không điều chỉnh checkpoint theo test.** '
        f"Toàn bộ 5.794 ảnh test gốc × 3 model = 17.382 dự đoán; bộ replication 100 foreground × 25 ảnh × 3 = 7.500 dự đoán. "
        '100 ảnh gốc của replication nằm trong full test; hai bảng lưu riêng và không được coi là mẫu độc lập bổ sung.\n\n'
        '## Điều kiện so sánh và giới hạn checkpoint\n\n'
        + '\n'.join('- '+s for s in audit['comparison_limitations'])+'\n\n'
        'Không tìm thấy ba image-backbone seed khớp nhau cho mỗi phương pháp. Các head có seed khác trong project '
        'không tương đương ba backbone train độc lập. Không suy ra seed variance từ composite hay bootstrap. '
        'ERM enhanced có augmentation, class-balanced sampling, ImageNet V2, 400 epoch, LR 0.001, weight decay 0; '
        'POTER seed1 dùng checkpoint validation-selected epoch258 từ 300 epoch, LR 1e-5, weight decay1. '
        'Do khác recipe/seed và thiếu liên kết lịch sử DPE–backbone bằng hash, kết quả là so sánh exploratory '
        'các checkpoint hiện có, chưa xác lập khác biệt do thuật toán POTER.\n\n'
        '## Audit và thiết kế\n\n'
        'Xem `phase0_audit.md` và `phase0_audit.json`: recompute trực tiếp 400 dòng, kiểm tra mask/foreground invariance, '
        'nhãn, số ảnh, loài, nền, duplicate. File được gọi `predictions(1).csv` không được cung cấp; '
        'audit dùng đúng run 400 dòng được lưu trước đó và nêu rõ thay thế này.\n\n'
        f"Seed chọn mẫu {config['sampling_seed']}, 50 foreground mỗi lớp, 10 loài mỗi lớp, 5 ảnh mỗi loài; "
        'chọn đều species rồi chọn ảnh, không dùng score/lỗi model. Loại 16 foreground sơ bộ và ba loài lỗi nhãn đã biết '
        'trước inference. `cohort.csv` và hash đã khóa trước lúc chạy. Đây là replication retrospective trên official test '
        'của checkpoint, không phải một tập test mới chưa từng được nghiên cứu.\n\n'
        'Hai nền Land và hai nền Water cố định dùng chung cho mọi foreground/model/severity. '
        'Gaussian sigma0/1.5/4/9; foreground/mask/crop cố định, normalized masked convolution chỉ lên RGB của chim. '
        'Không tạo ảnh bằng AI. 1.600 natural composites + 400 neutral foreground-only + 400 background-only '
        '(chỉ 4 nền duy nhất lặp theo foreground) + 100 ảnh gốc. '
        'Mask không đổi giữa severity; cả ba nhận cùng tensor SHA-256. '
        'Giữ label canonical 0=Landbird, 1=Waterbird. Neutral có semantic accuracy nhưng không có four-group WGA; '
        'background-only không có target chim/accuracy/WGA.\n\n'
        '## Toàn bộ test gốc — nhãn official\n\n'
        +table(orig,['model_name','n','semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy'])+'\n\n'
        'Giá trị accuracy/WGA ở các bảng tính theo %. Full inference mới có5.330 ảnh POTER đúng; '
        'log Kaggle epoch258 ghi5.331. WGA và3/4 group accuracy khớp; nhóm Landbird/Land chênh1 ảnh. '
        'Chưa xác định nguyên nhân (runtime và nguồn JPEG/cache cũ khác); không sửa dự đoán để khớp log. '
        'Acceptance test so với POTER Runner hiện tại vẫn qua. Chi tiết trong `historical_poter_log_comparison.json`.\n\n'
        '### Nhạy cảm với lỗi nhãn species\n\n'
        'Ba species Eastern Towhee, Western Meadowlark, Western Wood-Pewee bị nhãn official Waterbird '
        'do match substring Tern trong Eastern/Western. Báo cáo official là chính; corrected đổi ba loài về Landbird '
        'và tính lại group, exclude bỏ ba loài. Không sửa dữ liệu gốc, không train hay chọn checkpoint lại.\n\n'
        +table(affected,['model_name','label_policy','n','semantic_accuracy','wga'])+'\n\n'
        '## Replication đa loài — Gaussian\n\n'
        +table(natural,['model_name','severity','semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy',
                       'predicted_waterbird_fraction','land_drop_pp','water_drop_pp'])+'\n\n'
        '## Bất định và so sánh paired\n\n'
        'Bootstrap percentile95%, 2.000 lần; resample 50 source foreground mỗi lớp có hoàn lại. '
        'Mọi background/severity/model của cùng foreground đi cùng nhau; WGA được tính lại trong mỗi resample. '
        'Bất định này có điều kiện trên bốn nền và các checkpoint cố định, không đo khả năng tổng quát sang nền khác '
        'hay seed train khác; tương quan species không được loại bỏ hoàn toàn bằng bootstrap ở mức ảnh. '
        'Class-asymmetric drop = drop Landbird − drop Waterbird. Số dương trong so sánh POTER−baseline '
        'nghĩa là POTER giảm mạnh hơn; CI cắt0 không cung cấp bằng chứng phân biệt rõ ở mẫu này.\n\n'
        +table(delta,['comparison','severity','metric','estimate','ci95_low','ci95_high'])+'\n\n'
        '## Kết luận từ dữ liệu\n\n')
    for model in MODELS:
        clean=natural[(natural.model_name==model)&(natural.severity=='clean')].iloc[0]
        strong=natural[(natural.model_name==model)&(natural.severity=='strong')].iloc[0]
        d=ci[(ci.model_name==model)&(ci.comparison=='within_model')&(ci.severity=='strong')&(ci.metric=='class_asymmetric_drop')].iloc[0]
        text+=(f"- **{model}:** Landbird {clean.landbird_accuracy:.2f}% → {strong.landbird_accuracy:.2f}%; "
            f"Waterbird {clean.waterbird_accuracy:.2f}% → {strong.waterbird_accuracy:.2f}%; "
            f"WGA {clean.wga:.2f}% → {strong.wga:.2f}%. Chênh lệch drop hai lớp "
            f"{100*d.estimate:.2f} pp, CI95% [{100*d.ci95_low:.2f}; {100*d.ci95_high:.2f}] pp.\n")
    species_wide=species.pivot(index=['model_name','true_y','species_id'],columns='severity',values='accuracy')
    for model in MODELS:
        s=species_wide.loc[(model,0)]
        text+=f"- {model}: accuracy Landbird giảm ở **{int((s.clean>s.strong).sum())}/10 loài Landbird** khi clean→strong (5 source foreground/loài).\n"
    def comparison(other,metric):
        row=ci[(ci.comparison=='POTER_minus_'+other)&(ci.metric==metric)&(ci.severity=='strong')].iloc[0]
        return f"{100*row.estimate:+.2f} pp, CI95% [{100*row.ci95_low:+.2f}; {100*row.ci95_high:+.2f}] pp"
    text+=('\n**Trả lời câu hỏi chính:** kiểu suy giảm bất đối xứng theo lớp tái hiện trên nhiều loài Landbird, '
        'nhưng cũng thấy rõ ở DPE. Với strong, chênh lệch class-asymmetric drop POTER−DPE là '
        +comparison('DPE_legacy','class_asymmetric_drop')+'. '
        'So với enhanced ERM, chênh lệch class-asymmetric drop là '
        +comparison('ERM_enhanced','class_asymmetric_drop')+'; chênh lệch WGA drop là '
        +comparison('ERM_enhanced','wga_drop')+'. '
        'Khi CI chứa0, chưa đủ bằng chứng phân biệt độ suy giảm ở mẫu này. '
        'Không thể kết luận POTER yếu hơn cả hai baseline trên mọi tiêu chí, hoặc lỗi là riêng của POTER. '
        'Các phát biểu này chỉ áp dụng cho checkpoint/recipe hiện có.\n\n'
        'Các bảng/CI cho biết collapse có tái hiện trong nhiều foreground và loài hay không; '
        'không quy kết POTER là nguyên nhân duy nhất. Original và clean recomposition là hai điều kiện khác nhau; '
        'xem `composition_artifacts.csv` và ảnh masks/composites trong `examples/` để đánh giá tác động ghép ảnh. '
        'Thay đổi từ clean sang blur giữ nguyên mask, nền và geometry, nhưng thí nghiệm không loại hết artifacts '
        'hay chứng minh cơ chế causal. Không có kiểm định đa so sánh; các CI là exploratory. '
        'Xác suất của các model chưa được calibration theo cùng thủ tục; nên ưu tiên các thay đổi paired '
        'trong model và accuracy/flip thay vì coi độ lớn confidence giữa model là tương đương.\n\n'
        '### Kiểm tra hình ảnh và yếu tố gây nhiễu còn lại\n\n'
        'Đã xem hai ví dụ định trước: mask giữ hình chim và hướng chim, nhưng biên có các dải mềm; '
        'ảnh654 có cấu trúc mảnh/occlusion qua chim; chim1283 nghiêng và chạm biên hình. '
        'Ảnh ghép đặt chim vào vị trí không nhất thiết tự nhiên. Nền Water được chọn có vật thể nhô khỏi mặt nước; '
        'các đối tượng/người/động vật trong nền chưa được sàng lọc. Vì vậy đây là đổi ảnh nền có ngữ cảnh, '
        'không phải một can thiệp chỉ thay thuộc tính trừu tượng Land/Water. '
        'Chỉ kiểm tra thủ công hai ảnh minh họa, chưa adjudicate mask của cả100 ảnh. '
        '`mask_quality_audit.csv` ghi diện tích mask trong đúng crop224 và lượng alpha còn lại sau crop '
        'cho100 foreground; sigma tuyệt đối giống nhau có thể tác động khác nhau theo kích thước chim. '
        'Không loại ảnh hay đổi severity sau khi xem dự đoán.\n\n'
        '## Hình và dữ liệu\n\n'
        '![Class accuracy](01_class_accuracy.png)\n\n![WGA](02_wga.png)\n\n'
        '![Waterbird probability](03_waterbird_probability.png)\n\n![Prediction transitions](04_flip_heatmap.png)\n\n'
        '![Isolation controls](05_isolation_controls.png)\n\n![Model comparison CI](06_model_difference_ci.png)\n\n'
        '![Composition inspection](07_composition_inspection.png)\n\n'
        '![Exact model inputs](08_exact_model_inputs.png)\n\n'
        '`predictions.csv` giữ 7.500 dòng, `original_test_predictions.csv` giữ 17.382 dòng. '
        'Các row giữ source/species/background/label/model/model seed/severity/sigma/probabilities/scores, '
        'mask/input/tensor hashes. DPE dùng log(mean head probability), không được coi là logit CNN để so sánh độ lớn '
        'margin giữa model. `paired_flips.csv` có flip clean→degraded và đổi Land→Water background riêng, '
        'kèm chiều flip và delta P(Waterbird).\n\n'
        'Lưu ảnh PNG cho hai foreground được định trước (mỗi lớp một ảnh), còn các ảnh khác lưu recipe/source/hash '
        'đủ tái tạo để tránh đầy ổ C. `group_metrics.csv`, `class_metrics.csv`, `bootstrap_ci.csv`, '
        '`metrics.csv`, `original_test_metrics.csv`, `original_test_label_sensitivity.csv`, `validation.json` '
        'và `checkpoint_audit.json` là dữ liệu máy đọc. `integrity.json` khóa file xuất cuối cùng.\n\n'
        '## Thời gian thực đo\n\n'
        +f"Full test gốc, ba model: **{timings['full_original_test']['elapsed_seconds']:.1f} giây**. "
        +f"Replication100, ba model: **{timings['replication100']['elapsed_seconds']:.1f} giây**. "
        'Không gồm thời gian tìm/audit checkpoint, smoke test và lập báo cáo. ERM/DPE dùng chung forward backbone; '
        'batch32 và ghi CSV trực tiếp nên nhanh hơn chạy ba lượt riêng và xuất mọi PNG trong UI.\n\n'
        '## Chạy lại\n\n```powershell\n'
        '.\\.venv\\Scripts\\python.exe scripts\\investigate_poter_failure.py --config configs/poter_failure.yaml\n'
        '.\\.venv\\Scripts\\python.exe scripts\\report_poter_failure.py --output results/poter_failure\n```\n\n'
        'Đổi `output_root` sang thư mục mới trong config trước khi chạy lại; chương trình không ghi đè run cũ. '
        'Chưa làm các experiment optional contrast/UrbanCars; ưu tiên audit và experiments1–5. '
        'Experiment5 chỉ hoàn thành CI foreground trên một checkpoint/method; xác nhận đa seed còn thiếu assets.\n')
    test_log=out/'tests.log'
    if test_log.exists():
        text+='\n## Kiểm tra tự động\n\n'+test_log.read_text(encoding='utf-8').strip().splitlines()[-1]+'\n'
    (out/'summary.md').write_text(text,encoding='utf-8')
    (out/'audit.md').write_text((out/'phase0_audit.md').read_text(encoding='utf-8')+
        '\n\n# Frozen checkpoint audit\n\n'+'\n'.join('- '+s for s in audit['comparison_limitations'])+
        '\n\nSmoke reference max DPE probability error1.1921e-7; repeat difference0. '
        'All three normalized tensor hashes are identical per image. See validation.json for factorial/input checks.\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results/poter_failure')
    args=parser.parse_args();out=ROOT/args.output
    config=yaml.safe_load((out/'experiment_config.yaml').read_text(encoding='utf-8'))
    frame=pd.read_csv(out/'predictions.csv')
    orig=pd.read_csv(out/'original_test_predictions.csv')
    cohort=pd.read_csv(out/'cohort.csv')
    all_fg=pd.read_csv(ROOT/config['data_root']/'foregrounds_manifest.csv')
    # Check inference-time committed hashes before using the files.
    import json
    timings=json.loads((out/'timings.json').read_text())
    for filename,key in [('predictions.csv','replication100'),('original_test_predictions.csv','full_original_test')]:
        if sha256(out/filename)!=timings[key]['csv_sha256']:raise ValueError('Inference output changed')
    checks=dict(replication=validate_predictions(frame,cohort),original_test=validate_predictions(orig,all_fg,original=True))
    # Independent file audit after inference: source metadata and all original JPEG bytes.
    from src.counterfactual.common import contained
    for row in all_fg.itertuples():
        if sha256(contained(ROOT/config['data_root'],row.source_image_path))!=row.source_sha256:
            raise ValueError('Canonical test JPEG changed since dataset audit')
    checks['all_original_source_files_sha256_verified']=len(all_fg)
    canonical=pd.read_csv(ROOT/config['data_root']/'metadata.csv')
    canonical=canonical[canonical.split==2].set_index('img_id').loc[all_fg.bird_source_id]
    if not np.array_equal(canonical.y.to_numpy(),all_fg.y.to_numpy()) or not np.array_equal(
            (2*canonical.y+canonical.place).to_numpy(),all_fg.original_group.to_numpy()):
        raise ValueError('Dataset manifest labels disagree with canonical metadata')
    checks['canonical_metadata_labels_and_groups_verified']=True
    summary,groups,classes=metric_tables(frame)
    summary.to_csv(out/'metrics.csv',index=False);groups.to_csv(out/'group_metrics.csv',index=False);classes.to_csv(out/'class_metrics.csv',index=False)
    original,original_groups,original_classes=metric_tables(orig)
    original.to_csv(out/'original_test_metrics.csv',index=False)
    original_groups.to_csv(out/'original_test_group_metrics.csv',index=False)
    original_classes.to_csv(out/'original_test_class_metrics.csv',index=False)
    history=pd.read_csv((ROOT/config['poter_checkpoint']).parent/'test.csv')
    history=history[history.epoch==258].iloc[0]
    group_comparison=[]
    for row in original_groups[original_groups.model_name=='POTER'].itertuples():
        g=2*row.true_y+(row.bg_type=='water')
        old_correct=int(round(history[f'avg_acc_group:{g}']*row.n))
        group_comparison.append(dict(group_id=g,n=row.n,current_correct=row.correct,
            historical_correct=old_correct,difference_correct=row.correct-old_correct))
    write_json(out/'historical_poter_log_comparison.json',dict(epoch=258,groups=group_comparison,
        current_correct=sum(r['current_correct'] for r in group_comparison),
        historical_correct=sum(r['historical_correct'] for r in group_comparison),
        interpretation='One image correctness discrepancy with historical Kaggle log; cause unresolved. Current Runner parity passes; no predictions altered.',
        historical_log_sha256=sha256((ROOT/config['poter_checkpoint']).parent/'test.csv')))
    sensitivity=pd.concat([metric_tables(orig,label_policy=p)[0] for p in ['official','corrected','exclude_known_errors']],ignore_index=True)
    sensitivity.to_csv(out/'original_test_label_sensitivity.csv',index=False)
    flips=paired_flips(frame);flips.to_csv(out/'paired_flips.csv',index=False)
    flips.groupby(['model_name','severity','pair_type','true_y']).agg(
        source_foregrounds=('source_image_id','nunique'),paired_images=('flip','size'),
        flip_rate=('flip','mean'),land_to_water_rate=('land_to_water','mean'),
        water_to_land_rate=('water_to_land','mean'),mean_delta_p_waterbird=('delta_p_waterbird','mean')).reset_index().to_csv(out/'flip_metrics.csv',index=False)
    natural=frame[frame.control=='natural'].copy();natural['correct']=natural.true_y==natural.pred_y
    species=natural.groupby(['model_name','true_y','species_id','severity']).agg(
        accuracy=('correct','mean'),source_foregrounds=('source_image_id','nunique'),images=('image_id','size'),
        mean_p_waterbird=('p_waterbird','mean')).reset_index()
    species.to_csv(out/'species_metrics.csv',index=False)
    ci=bootstrap_comparison(frame,samples=config['bootstrap_samples'],seed=config['sampling_seed'])
    ci.to_csv(out/'bootstrap_ci.csv',index=False)
    # Bird-matched original-to-clean composition reference, never pooled with blur.
    original_ref=frame[frame.control=='original'][['model_name','source_image_id','pred_y','p_waterbird','true_y']]
    clean=frame[(frame.control=='natural')&(frame.severity=='clean')].merge(original_ref,on=['model_name','source_image_id'],suffixes=('','_original'),validate='many_to_one')
    clean['composition_flip']=clean.pred_y!=clean.pred_y_original
    clean['delta_p_waterbird']=clean.p_waterbird-clean.p_waterbird_original
    clean[['model_name','source_image_id','species_id','true_y','background_id','pred_y_original','pred_y','composition_flip','delta_p_waterbird']].to_csv(out/'composition_artifacts.csv',index=False)
    plots(out,summary,classes,flips,ci)
    inspection_figure(out)
    checks['mask_and_example_inspection']=inspect_saved_inputs(out,frame,config)
    write_json(out/'validation.json',checks)
    report(out,summary,classes,ci,original,sensitivity,checks,config,species)
    import platform,torch,torchvision,matplotlib
    from datetime import datetime,timezone
    write_json(out/'analysis_provenance.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,python=platform.python_version(),torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),pandas=pd.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__,
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
            ROOT/'src/counterfactual/failure_statistics.py',ROOT/'tests/test_failure_investigation.py']},
        tests_log_sha256=sha256(out/'tests.log') if (out/'tests.log').exists() else None,
        checkpoint_and_input_provenance='checkpoint_audit.json; experiment_config.yaml; timings.json; validation.json',
        inference_predictions_changed=False))
    write_json(out/'status.json',dict(status='completed',inference_rows=len(frame)+len(orig),
        statistical_unit='source foreground; class-stratified paired bootstrap',model_seeds_per_method=1))
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(original[['model_name','semantic_accuracy','wga']].to_string(index=False))
    print(summary[summary.control=='natural'][['model_name','severity','semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy']].to_string(index=False))


if __name__=='__main__':main()
