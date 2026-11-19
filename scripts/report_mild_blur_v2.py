"""Metrics, clustered statistics and the final paired mild-blur report."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from src.counterfactual.common import sha256,write_json,write_csv
from src.counterfactual.failure_statistics import (validate_predictions,metric_tables,paired_flips,
    bootstrap_comparison,stratified_draws)
from src.counterfactual.investigation import MODELS

LEVELS=['clean','mild','moderate','strong','extreme']
MAIN=LEVELS[:4]
SIGMAS=dict(zip(LEVELS,[0.,.5,1.,2.,4.]))
NAMES={'POTER':'POTER','ERM_enhanced':'ERM enhanced','DPE_legacy':'DPE legacy'}
COLORS={'POTER':'#b64b39','ERM_enhanced':'#2878a4','DPE_legacy':'#437e4c'}


def neutral_bootstrap(frame,samples,seed):
    data=frame[frame.control=='foreground_only'].copy()
    ids=sorted(data.source_image_id.unique())
    labels=data.drop_duplicates('source_image_id').set_index('source_image_id').loc[ids].true_y.to_numpy().astype(int)
    draws=stratified_draws(labels,samples,seed)
    direct=[np.flatnonzero(labels==y)[None,:] for y in (0,1)]
    rows=[]
    for model in MODELS:
        for level in LEVELS:
            d=data[(data.model_name==model)&(data.severity==level)].set_index('source_image_id').loc[ids]
            a=np.column_stack([(d.pred_y==d.true_y).to_numpy().astype(float),d.p_waterbird.to_numpy(),(d.pred_y==1).to_numpy().astype(float)])
            def calculate(indices):
                c0,c1=[a[i].mean(1) for i in indices]
                return dict(accuracy=(c0[:,0]+c1[:,0])/2,landbird_accuracy=c0[:,0],waterbird_accuracy=c1[:,0],
                    p_waterbird_landbird=c0[:,1],p_waterbird_waterbird=c1[:,1],
                    predicted_waterbird_given_landbird=c0[:,2],predicted_waterbird_given_waterbird=c1[:,2])
            estimate,distribution=calculate(direct),calculate(draws)
            for metric,values in distribution.items():
                lo,hi=np.quantile(values,[.025,.975])
                rows.append(dict(model_name=model,comparison='within_model',severity=level,metric=metric,
                    estimate=float(estimate[metric][0]),ci95_low=float(lo),ci95_high=float(hi),bootstrap_samples=samples,
                    bootstrap_seed=seed,source_foregrounds=len(ids),unit='source_image_id; class-stratified',
                    seed_level_std=None,seed_status='one fixed checkpoint; no training-seed uncertainty',condition='foreground_only'))
    return pd.DataFrame(rows)


def shared_comparison(frame,old):
    keys=['model_name','source_image_id','background_id','control','sigma']
    shared=frame.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if len(shared)!=4500:raise ValueError('Incorrect old/new shared sigma coverage')
    for field in ['foreground_id','species_id','source_y','original_group','mask_sha256','source_sha256',
                  'raw_input_sha256','model_input_sha256','normalized_tensor_sha256','pred_y','p_landbird','p_waterbird',
                  'logit_landbird','logit_waterbird','model_seed','sampling_seed','source_split']:
        if not np.array_equal(shared[field+'_new'].to_numpy(),shared[field+'_old'].to_numpy()):
            raise ValueError('Shared previous/current prediction mismatch: '+field)
    result=shared.groupby(['model_name','control','sigma','severity_old','severity_new']).agg(
        matched_rows=('source_image_id','size'),source_foregrounds=('source_image_id','nunique')).reset_index()
    result['raw_hash_identical']=True;result['tensor_hash_identical']=True;result['prediction_identical']=True
    result['execution']='main rows reuse exact cached outputs; separate shared-sigma smoke independently reruns selected images'
    return result


def background_table(frame,flips):
    natural=frame[frame.control=='natural'];paired=flips[flips.pair_type=='land_background_to_water_background']
    output=[]
    for model in MODELS:
        for level in LEVELS:
            for scope,y in [('all',None),('landbird',0),('waterbird',1)]:
                d=natural[(natural.model_name==model)&(natural.severity==level)]
                pairs=paired[(paired.model_name==model)&(paired.severity==level)]
                if y is not None:d=d[d.true_y==y];pairs=pairs[pairs.true_y==y]
                land,water=[d[d.bg_type==bg] for bg in ('land','water')]
                output.append(dict(model_name=model,severity=level,sigma=SIGMAS[level],scope=scope,
                    source_foregrounds=d.source_image_id.nunique(),background_swap_pairs=len(pairs),
                    accuracy_land_background=float((land.true_y==land.pred_y).mean()),
                    accuracy_water_background=float((water.true_y==water.pred_y).mean()),
                    background_swap_flip_rate=float(pairs.flip.mean()),
                    mean_abs_delta_p_waterbird=float(pairs.delta_p_waterbird.abs().mean()),
                    mean_signed_delta_p_waterbird=float(pairs.delta_p_waterbird.mean())))
    return pd.DataFrame(output)


def figures(out,summary,classes,groups,flips):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white'})
    xs=np.array([SIGMAS[l] for l in LEVELS])
    def line(ax,values,label,color,style='-o'):
        values=np.asarray(values,dtype=float)
        ax.plot(xs[:4],values[:4],style,label=label,color=color,lw=1.8,markersize=5)
        ax.plot(xs[3:],values[3:],'--',color=color,alpha=.6,lw=1.2)
        ax.plot(xs[-1],values[-1],marker='D',markerfacecolor='white',markeredgecolor=color,linestyle='none')
    def axes(ax,ylabel,title):
        ax.set(xlabel='Foreground Gaussian sigma (model-view pixels)',ylabel=ylabel,title=title,xticks=xs,xlim=(-.12,4.2))
        ax.axvspan(2.2,4.2,color='#8794a0',alpha=.08)
        ax.text(.75,.85 if ylabel=='Prediction flip rate (%)' else .02,
            'σ=4 supplementary',transform=ax.transAxes,fontsize=8,color='#586673')
        ax.grid(axis='y',alpha=.18)
    def save(fig,name):
        fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
    for column,title,filename in [('semantic_accuracy','Average accuracy','accuracy_vs_blur.png'),('wga','Worst-group accuracy','wga_vs_blur.png')]:
        fig,ax=plt.subplots(figsize=(8.5,5),layout='constrained')
        for model in MODELS:
            d=summary[(summary.model_name==model)&(summary.control=='natural')].set_index('severity').loc[LEVELS]
            line(ax,100*d[column],NAMES[model],COLORS[model])
        axes(ax,'Accuracy (%)',title+' on matched natural composites');ax.set_ylim(0,103);ax.legend()
        save(fig,filename)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained',sharey=True)
    for ax,model in zip(axs,MODELS):
        for y,label,color in [(0,'Landbird','#b86e31'),(1,'Waterbird','#247fa3')]:
            d=classes[(classes.model_name==model)&(classes.control=='natural')&(classes.true_y==y)].set_index('severity').loc[LEVELS]
            line(ax,100*d.accuracy,label,color)
        axes(ax,'Accuracy (%)',NAMES[model]);ax.set_ylim(0,103);ax.legend()
    fig.suptitle('Class-specific accuracy: same 50 foregrounds per class at every sigma')
    save(fig,'class_accuracy_vs_blur.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharey=True)
    for (y,bg),ax in zip([(0,'land'),(0,'water'),(1,'land'),(1,'water')],axs.flat):
        for model in MODELS:
            d=groups[(groups.model_name==model)&(groups.control=='natural')&(groups.true_y==y)&(groups.bg_type==bg)].set_index('severity').loc[LEVELS]
            line(ax,100*d.accuracy,NAMES[model],COLORS[model])
        axes(ax,'Accuracy (%)',['Landbird','Waterbird'][y]+' on '+bg);ax.set_ylim(0,103);ax.legend(fontsize=8)
    save(fig,'group_accuracy_vs_blur.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for y,ax in enumerate(axs):
        for model in MODELS:
            d=classes[(classes.model_name==model)&(classes.control=='natural')&(classes.true_y==y)].set_index('severity').loc[LEVELS]
            line(ax,d.mean_p_waterbird,NAMES[model],COLORS[model])
        axes(ax,'Mean P(Waterbird)','True '+['Landbird','Waterbird'][y]);ax.set_ylim(0,1);ax.legend(fontsize=8)
        ax.axhline(.5,color='gray',linestyle=':',lw=.8)
    save(fig,'waterbird_probability_vs_blur.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for y,ax in enumerate(axs):
        for model in MODELS:
            d=flips[(flips.model_name==model)&(flips.pair_type=='clean_to_degraded')&(flips.true_y==y)].groupby('severity').flip.mean().loc[LEVELS]
            line(ax,100*d,NAMES[model],COLORS[model])
        axes(ax,'Prediction flip rate (%)','True '+['Landbird','Waterbird'][y]);ax.set_ylim(0,103);ax.legend(fontsize=8)
    save(fig,'prediction_flip_rates.png')
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained',sharey=True)
    for model,ax in zip(MODELS,axs):
        for control,label,color in [('natural','Natural composite','#2878a4'),('foreground_only','Foreground on gray','#b86e31')]:
            d=summary[(summary.model_name==model)&(summary.control==control)].set_index('severity').loc[LEVELS]
            line(ax,100*d.semantic_accuracy,label,color)
        axes(ax,'Average accuracy (%)',NAMES[model]);ax.set_ylim(0,103);ax.legend(fontsize=8)
    save(fig,'foreground_only_vs_composite.png')


def mdtable(headers,rows):
    escape=lambda value:str(value).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(escape,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(escape,row))+' |' for row in rows])


def pct(value):return f'{100*float(value):.2f}%'
def pp(value):return f'{0. if abs(float(value))<1e-10 else 100*float(value):+.2f}'


def final_report(out,config,summary,classes,drops,ci,bg,validation,previous_metrics):
    timings=json.loads((out/'timings.json').read_text());qc=json.loads((out/'qc_audit.json').read_text())
    checkpoints=json.loads((out/'checkpoint_audit.json').read_text(encoding='utf-8'))
    natural=summary[summary.control=='natural']
    def matrix(field):
        return mdtable(['Model','Clean σ0','Mild σ0,5','Moderate σ1','Strong σ2','Extreme σ4 (bổ sung)'],
            [[NAMES[m],*[pct(natural[(natural.model_name==m)&(natural.severity==l)].iloc[0][field]) for l in LEVELS]] for m in MODELS])
    def interval(model,level,metric,comparison='within_model'):
        r=ci[(ci.condition=='natural')&(ci.model_name==model)&(ci.severity==level)&(ci.metric==metric)&(ci.comparison==comparison)].iloc[0]
        return f"{pp(r.estimate)} [{pp(r.ci95_low)}; {pp(r.ci95_high)}]"
    text=('# Waterbirds mild blur v2: POTER, ERM enhanced và DPE legacy\n\n'
        '## 1. Execution summary\n\n'
        '**Đã hoàn thành đánh giá thật bằng checkpoint có sẵn, không train lại hoặc sửa model.** '
        'Thay bộ sigma cũ0/1,5/4/9 bằng **0/0,5/1/2** cho phân tích chính; **4** chỉ là diagnostic bổ sung. '
        'Giữ nguyên100 foreground,20 loài (50 ảnh và10 loài/lớp,5 ảnh/loài), bốn nền, pairing, mask, source test và seed20261118.\n\n'
        '2.000 natural composites +500 foreground-only +400 background assignments +100 ảnh source gốc '
        '= **3.000 inputs/model và9.000 dòng dự đoán cho ba model**. '
        f"Có {validation['unique_raw_images']} ảnh raw duy nhất; background-only chỉ có4 nền lặp theo foreground. "
        f"**{timings['new_prediction_rows']:,} dòng inference mới** ở sigma0,5/1/2; "
        f"**{timings['reused_prediction_rows']:,} dòng tái sử dụng** có checkpoint, raw pixel, crop và normalized tensor hash khớp. "
        'Sáu ảnh smoke test được chạy trực tiếp/lặp để kiểm tra parity, tách khỏi9.000 dòng chính. '
        f"Vòng đánh giá có cache mất {timings['elapsed_seconds']:.2f} giây, chưa gồm QC, load model, smoke và lập báo cáo.\n\n"
        +mdtable(['Model','Model seed','Checkpoint','SHA256 (rút gọn)'],[
            ['POTER',1,'best_validation.pt, epoch258/300',checkpoints['POTER']['checkpoint_sha256'][:16]],
            ['ERM enhanced',0,'ckpt_last.pt, enhanced/class-balanced ERM',checkpoints['ERM_enhanced']['checkpoint_sha256'][:16]],
            ['DPE legacy',0,'15 heads; cùng backbone ERM enhanced',checkpoints['DPE_legacy']['heads_sha256'][:16]]])+'\n\n'
        'Mọi SHA256 đầy đủ và giới hạn lịch sử nằm trong `checkpoint_audit.json`. ERM không phải uniform ERM tiêu chuẩn; '
        'seed/recipe chưa khớp POTER. DPE legacy thiếu hash liên kết backbone từ lúc huấn luyện; '
        'đây là cặp lưu cùng thư mục được khóa hiện tại, chưa xác minh độc lập association lịch sử. '
        'Head-training IDs khớp1.199 validation IDs canonical, không có test.\n\n'
        'Nguồn pipeline: `src/counterfactual/compositor.py` (foreground_layer, composite), '
        '`src/counterfactual/investigation.py` (FrozenComparison, intervention_records, evaluate_stream), '
        '`src/counterfactual/model.py` và `src/datasets/waterbirds.py` (Runner/preprocess). '
        'CSV trước dùng canonical `results/poter_failure/predictions.csv`,34 cột, do '
        '`scripts/investigate_poter_failure.py` tạo. `predictions(2).csv` không được đính kèm; '
        'không giả định một bản đổi tên ngoài repository đã được kiểm tra. File mới giữ nguyên cả34 cột và thứ tự.\n\n'
        '## 2. Image quality\n\n'
        'Foreground được premultiply alpha rồi resize bilinear về256×256. Gaussian sau resize, '
        'trước ghép nền và center crop224×224. Vì crop không đổi scale, **sigma là pixel trong đúng input224 của model**. '
        'Preprocess resize256, crop224, RGB/ImageNet normalization và nội suy giữ nguyên. '
        'Normalized convolution blur RGB×alpha / blur alpha giữ alpha cố định, tránh trộn RGB đen của vùng trong suốt. '
        'Nền không blur. Không dùng ảnh generative AI.\n\n'
        'Contact sheet chọn deterministically1 foreground/loài từ cohort:10 Landbird,10 Waterbird,20 loài. '
        'Mỗi foreground có một background cố định qua năm cột224×224. Contact sheet foreground-only '
        'gồm ba ảnh mỗi lớp trên xámRGB127. Đã kiểm tra dimensions, alpha, invariance và pairing trên toàn3.000 inputs '
        '**trước khi load model/inference**. Clean vàsigma4 khớp hash thí nghiệm trước.\n\n'
        'Quan sát contact sheet vẫn thấy một số biên mask, occluder hoặc mảnh màu/nền nằm trong cutout '
        '(ví dụ vùng trên ảnh2599 và vùng quanh chân/đuôi ảnh1715). Các yếu tố này có sẵn ở clean và được giữ nguyên; '
        'không loại ảnh sau khi xem model. Nền có thể chứa người/động vật/vật thể khác. '
        '**Chưa có independent human recognizability validation**; kiểm tra tự động hay quan sát của trợ lý '
        'không xác lập ngưỡng bảo toàn nhãn semantic. Các sigma chỉ là mức ứng viên.\n\n'
        '![20 foreground QC](blur_contact_sheet.png)\n\n![Foreground-only QC](foreground_only_contact_sheet.png)\n\n'
        '## 3. Main results\n\n'
        'Mỗi natural severity/model có400 composites,200/lớp và100/nhóm lớp×nền. '
        'AA là tỷ lệ đúng trên toàn bộ400 ảnh. WGA là min của bốn group accuracy. '
        'Clean là clean recomposition trên cùng bốn nền; không đồng nhất với ảnh source gốc.\n\n'
        '### WGA\n\n'+matrix('wga')+'\n\n### Average accuracy\n\n'+matrix('semantic_accuracy')+'\n\n'
        '### Accuracy từng lớp\n\n'
        +mdtable(['Model','Lớp','σ0','σ0,5','σ1','σ2','σ4 (bổ sung)'],[
            [NAMES[m],label,*[pct(classes[(classes.model_name==m)&(classes.control=='natural')&(classes.true_y==y)&(classes.severity==l)].iloc[0].accuracy) for l in LEVELS]]
            for m in MODELS for y,label in [(0,'Landbird'),(1,'Waterbird')]])+'\n\n'
        'Ảnh source gốc100 và background-only được giữ nguyên. Neutral có nhãn chim và semantic accuracy, '
        'nhưng không có WGA bốn nhóm; background-only không có target chim, accuracy hoặc WGA. '
        'Chi tiết trong metrics.csv/group_metrics.csv/class_metrics.csv.\n\n'
        '![AA](accuracy_vs_blur.png)\n\n![WGA](wga_vs_blur.png)\n\n'
        '![Class accuracy](class_accuracy_vs_blur.png)\n\n![Groups](group_accuracy_vs_blur.png)\n\n'
        '## 4. Class-asymmetric robustness analysis\n\n'
        'Drop = clean−corrupted. Asymmetry = Drop Landbird−Drop Waterbird. '
        'Raw class gap = Waterbird accuracy−Landbird accuracy, bao gồm chênh lệch có sẵn ở clean. '
        'Các số dưới là điểm phần trăm (pp).\n\n'
        +mdtable(['Model','Sigma','Drop WGA','Drop Land','Drop Water','Asymmetry [CI95%]','Raw class gap'],[
            [NAMES[r.model_name],r.sigma,pp(r.wga_drop),pp(r.landbird_accuracy_drop),pp(r.waterbird_accuracy_drop),
             interval(r.model_name,r.severity,'class_asymmetric_drop'),pp(r.class_accuracy_gap)]
            for r in drops[drops.severity.isin(MAIN[1:])].itertuples()])+'\n\n'
        'Sigma4 không tham gia kết luận chính; các giá trị/CIs bổ sung nằm trong các CSV với severity=extreme. '
        'Bootstrap2.000 lần, class-stratified source foreground, cùng resampling indices cho các model/severity. '
        'Bốn background của cùng foreground đi cùng nhau; WGA được tính lại trong từng resample. '
        'CI phản ánh sampling variability theo foreground, **không** phản ánh checkpoint/training-seed uncertainty '
        'hay một tập backgrounds mới. Tương quan giữa các ảnh cùng species còn là hạn chế.\n\n'
        '### Chênh lệch POTER−baseline trong phân tích chính\n\n'
        +mdtable(['So sánh','Sigma','Chênh Drop WGA [CI95%]','Chênh Asymmetry [CI95%]'],[
            ['POTER−'+NAMES[other],SIGMAS[level],interval('POTER',level,'wga_drop','POTER_minus_'+other),
             interval('POTER',level,'class_asymmetric_drop','POTER_minus_'+other)]
            for other in MODELS[1:] for level in MAIN[1:]])+'\n\n'
        'Số dương: POTER giảm mạnh hơn baseline theo chỉ số đó. CI chứa0: chưa phân biệt rõ ở sample hiện tại. '
        'Không hiệu chỉnh nhiều so sánh; đây là exploratory comparisons của các checkpoint/recipe hiện có.\n\n'
        '### Prediction bias ở mức mild σ0,5\n\n'
        +mdtable(['Model','Lớp thật','P(Water) clean','P(Water) σ0,5','Pred Water clean','Pred Water σ0,5'],[
            [NAMES[m],label,*[pct(classes[(classes.model_name==m)&(classes.control=='natural')&(classes.true_y==y)&(classes.severity==l)].iloc[0][field])
                for field in ['mean_p_waterbird','predicted_waterbird_fraction'] for l in ['clean','mild']]]
            for m in MODELS for y,label in [(0,'Landbird'),(1,'Waterbird')]])+'\n\n'
        'DPE score là log(mean head probability), không phải native CNN logit. Xác suất giữa các model '
        'chưa calibration chung. Không diễn giải xác suất tuyệt đối như bằng chứng model này tin cậy hơn model khác. '
        'Phân tích hướng flip và bias đầy đủ theo sigma/lớp nằm trong prediction_bias.csv, flip_metrics.csv và paired_flips.csv.\n\n'
        '![Conditional probability](waterbird_probability_vs_blur.png)\n\n![Flips](prediction_flip_rates.png)\n\n'
        '## 5. Background dependence analysis\n\n'
        'Giữ cùng foreground/severity và ghép Land/Water theo hai pair đã khóa. '
        'Swap flip là tỷ lệ đổi nhãn dự đoán; mean absolute ΔP là trung bình |P(Water|Water background)−P(Water|Land background)|. '
        'Hai cột cuối dùng% hoặc pp như ghi trong header.\n\n'
        +mdtable(['Model','Sigma','Acc nền Land','Acc nền Water','Swap flip (%)','Mean |ΔP| (pp)'],[
            [NAMES[r.model_name],r.sigma,pct(r.accuracy_land_background),pct(r.accuracy_water_background),
             pct(r.background_swap_flip_rate),f'{100*r.mean_abs_delta_p_waterbird:.2f}']
            for r in bg[(bg.scope=='all')&bg.severity.isin(MAIN)].itertuples()])+'\n\n'
        'Có thay đổi dự đoán khi đổi ảnh nền là bằng chứng nhạy cảm với context trong các cặp đã chọn. '
        'Bốn nền chứa nhiều yếu tố ngoài Land/Water; không suy ra một cơ chế causal chỉ từ việc flip. '
        'Background-only chỉ đo prediction tendency trên4 ảnh OOD, không có accuracy hay số lượng background độc lập400. '
        'CI của background-swap flip và mean absolute ΔP ở bootstrap_ci.csv; class-conditioned kết quả ở background_sensitivity.csv.\n\n'
        '### Đối chiếu với foreground-only trên nền xám\n\n'
        +mdtable(['Model','Sigma','AA composite','AA foreground-only','Land composite','Land foreground-only'],[
            [NAMES[m],SIGMAS[l],
             pct(summary[(summary.model_name==m)&(summary.control=='natural')&(summary.severity==l)].iloc[0].semantic_accuracy),
             pct(summary[(summary.model_name==m)&(summary.control=='foreground_only')&(summary.severity==l)].iloc[0].semantic_accuracy),
             pct(summary[(summary.model_name==m)&(summary.control=='natural')&(summary.severity==l)].iloc[0].landbird_accuracy),
             pct(summary[(summary.model_name==m)&(summary.control=='foreground_only')&(summary.severity==l)].iloc[0].landbird_accuracy)]
            for m in MODELS for l in ['mild','strong']])+'\n\n'
        'Đây là so sánh point estimates trên cùng100 foreground, nhưng natural có4 nền/foreground còn neutral có1. '
        'POTER ởsigma2 có Landbird accuracy cao hơn khi đặt trên xám so với bốn nền đã chọn; '
        'hướng chênh không đồng nhất giữa các model. Điều này cho thấy việc xem isolation control là cần thiết, '
        'chưa chứng minh nguyên nhân hay khái quát sang background mới. CIs riêng của neutral trong bootstrap_ci.csv, '
        'không dùng chúng như một CI paired cho chênh neutral−composite.\n\n'
        '![Foreground isolation](foreground_only_vs_composite.png)\n\n'
        '## 6. Comparison with previous experiment\n\n'
        'So sánh theo **sigma thực**, không theo tên mild/moderate/strong: mild cũ=1,5 còn mild mới=0,5; '
        'moderate cũ=4 là extreme mới. Không dùng chúng làm cùng điều kiện. '
        'Shared settingssigma0 và4 có raw input/model crop/normalized tensor hashes giống nhau trên mọi foreground/nền/model. '
        'Các row này tái sử dụng đúng xác suất/log-score cũ sau xác minh hash; đây không phải rerun độc lập toàn bộ shared rows. '
        'Ngoài cache,12 model-image rows sharedsigma trong smoke được inference trực tiếp để kiểm tra numerical parity.\n\n'
        +mdtable(['Model','Sigma chung','Tên cũ','Tên mới','WGA cũ','WGA mới'],[
            [NAMES[m],sigma,oldlevel,newlevel,
             pct(previous_metrics[(previous_metrics.model_name==m)&(previous_metrics.control=='natural')&(previous_metrics.severity==oldlevel)].iloc[0].wga),
             pct(natural[(natural.model_name==m)&(natural.severity==newlevel)].iloc[0].wga)]
            for m in MODELS for sigma,oldlevel,newlevel in [(0,'clean','clean'),(4,'moderate','extreme')]])+'\n\n'
        'shared_sigma_reproducibility.csv giữ đối chiếu đầy đủ; shared_sigma_inference_smoke.csv ghi sai số rerun thực. '
        'Các điểm cũsigma1,5 và9 được giữ nguyên ở thư mục cũ và không trộn vào đường cong chính mới. '
        'Hiệu ứng lớn ởsigma9 trước đây không thể dùng làm bằng chứng rằng blur0,5 cũng gây collapse. '
        'Phân tích mới kiểm tra trực tiếp vùngsigma thấp, tách giảm accuracy nhỏ ở0,5 khỏi đáp ứng lệch lớp rõ hơn ở1–2.\n\n'
        '## 7. Research implications\n\n'
        '**Ở mild σ0,5:** CI asymmetry trong từng model đều chạm/chứa0 ở mẫu này; '
        'chưa đủ bằng chứng để gọi suy giảm lệch về Landbird rõ ràng ở cả ba. '
        'POTER có raw class gap ở clean; phần chênh sẵn này không được tính như degradation-induced asymmetry.\n\n'
        '**POTER so với baseline:** bảng paired cho thấy ở0,5 chênh Drop WGA POTER−DPE là '
        +interval('POTER','mild','wga_drop','POTER_minus_DPE_legacy')+' pp, '
        'nhưng chênh asymmetry POTER−DPE là '
        +interval('POTER','mild','class_asymmetric_drop','POTER_minus_DPE_legacy')+' pp. '
        'Có bằng chứng chênh WGA-drop trong bộ checkpoint này ở0,5, chưa đủ bằng chứng khác biệt về asymmetry với DPE. '
        'Ở2, POTER có WGA-drop ít hơn enhanced ERM (xem CI âm trong bảng), dù đáp ứng lệch lớp lớn hơn ERM. '
        'Không kết luận POTER yếu hơn mọi baseline theo mọi tiêu chí.\n\n'
        '**Bias khi blur nhẹ:** bảng conditional prediction cho thấy các thay đổi nhỏ ở0,5 và bias có sẵn ởclean. '
        'Đó là mô tả xác suất/nhãn dự đoán; không chứng minh class preference là cơ chế causal. '
        'Ở1–2, CIs asymmetry của POTER và DPE nằm trên0; hiệu ứng không chỉ xuất hiện ởPOTER.\n\n')
    for level in MAIN[1:]:
        text+=f'### Sigma {SIGMAS[level]:g}\n\n'
        for model in MODELS:
            r=ci[(ci.condition=='natural')&(ci.model_name==model)&(ci.comparison=='within_model')&
                (ci.severity==level)&(ci.metric=='class_asymmetric_drop')].iloc[0]
            evidence='CI nằm trên0, hỗ trợ suy giảm lệch về Landbird trong mẫu này' if r.ci95_low>1e-10 else (
                'CI nằm dưới0, suy giảm lệch về Waterbird trong mẫu này' if r.ci95_high< -1e-10 else 'CI chứa0, chưa phân biệt rõ mức suy giảm giữa hai lớp')
            text+=f'- **{NAMES[model]}:** asymmetry {interval(model,level,"class_asymmetric_drop")} pp; {evidence}.\n'
        text+='\n'
    text+=('Mức độ POTER dễ bị tổn thương hơn baseline phải đọc từ **chênh lệch drop paired và CI**, '
        'không suy ra chỉ từ accuracy tuyệt đối ở một severity. Các bảng trên tách WGA-drop và asymmetry; '
        'một model có thể lệch lớp hơn nhưng không mất WGA nhiều hơn. '
        'Prediction bias và sự thay đổi theo background là mô tả đầu ra có điều kiện; chưa xác nhận class prior '
        'hay background reliance là cơ chế gây ra suy giảm. Không điều chỉnh sigma/cohort/checkpoint sau khi xem kết quả.\n\n'
        '## 8. Limitations and next steps\n\n'
        '- Một checkpoint/method, seed và training recipe không khớp hoàn toàn. Không có CI training-seed.\n'
        '- Cohort100 là existing official test đã được nghiên cứu, không phải held-out cohort mới độc lập.\n'
        '- Chưa có independent human recognizability validation; sigma0,5/1/2 không phải ngưỡng semantic đã xác nhận.\n'
        '- Mask, occlusion, center crop, vị trí ghép và background objects có thể gây artifacts.\n'
        '- Chỉ20 loài,5 foreground/loài và4 nền được reuse; không khái quát CI sang nền/species mới.\n'
        '- DPE legacy thiếu hash liên kết backbone lịch sử; current pair và checkpoint hashes được khóa, nhưng association cũ chưa được chứng minh.\n'
        '- CI exploratory chưa hiệu chỉnh multiple comparisons. Không dùng test cohort để train hoặc tune hyperparameters.\n\n'
        'Bước tiếp theo: nhận đánh giá độc lập của con người trên contact sheet trước khi gọi can thiệp bảo toàn semantic; '
        'kiểm tra mask có occluder/mảnh nền; dùng cohort/background mới đã khóa và checkpoint đa seed khớp recipe nếu có. '
        'Giữ nguyên mọi kết quả hiện tại.\n\n'
        '### Tái lập và kiểm tra\n\n```powershell\n'
        '.\\.venv\\Scripts\\python.exe scripts\\run_mild_blur_v2.py --stage qc\n'
        '.\\.venv\\Scripts\\python.exe scripts\\run_mild_blur_v2.py --stage infer\n'
        '.\\.venv\\Scripts\\python.exe scripts\\run_mild_blur_v2.py --stage report\n```\n\n'
        'Chạy mới bằng output_root mới trong configs/mild_blur_v2.yaml. Không ghi đè predictions cũ. '
        'Bản output này có status completed; integrity.json khóa file, sourcecode hashes trong blur_config.json/analysis_provenance.json. '
        'Các kiểm tra counts/duplicates/missing/label/tensor/group coverage nằm trong validation.json.\n')
    text+=('\n### Các file kết quả\n\n'
        '- `predictions.csv`: 9.000 dòng, nguyên34 cột của schema trước.\n'
        '- `metrics.csv`, `group_metrics.csv`, `class_metrics.csv`, `degradation_metrics.csv`, `prediction_bias.csv`.\n'
        '- `paired_flips.csv`, `flip_metrics.csv`, `background_sensitivity.csv`, `bootstrap_ci.csv`.\n'
        '- `blur_contact_sheet.png`, `foreground_only_contact_sheet.png`, `blur_config.json`, `qc_foregrounds.csv`.\n'
        '- `accuracy_vs_blur.png`, `wga_vs_blur.png`, `class_accuracy_vs_blur.png`, `group_accuracy_vs_blur.png`.\n'
        '- `waterbird_probability_vs_blur.png`, `prediction_flip_rates.png`, `foreground_only_vs_composite.png`.\n'
        '- `image_integrity.csv`, `qc_audit.json`, `shared_sigma_reproducibility.csv`, `shared_sigma_inference_smoke.csv`.\n'
        '- `prediction_reuse_audit.csv`, `cohort.csv`, `checkpoint_audit.json`, `smoke_test.json`, `timings.json`.\n'
        '- `validation.json`, `analysis_provenance.json`, `status.json`, `tests.log`, `integrity.json`, `report.md`.\n'
        '- `examples/`: ảnh/mask và30 ảnh đầu vào cho mỗi một trong hai foreground định trước.\n\n'
        '`files_manifest.csv` liệt kê chính xác mọi file sản phẩm, kích thước và SHA256 '
        '(trừ chính manifest và integrity.json để tránh vòng hash).\n')
    if (out/'tests.log').exists():
        text+='\nKết quả kiểm tra tự động: '+(out/'tests.log').read_text(encoding='utf-8').strip().splitlines()[-1]+'\n'
    browser=ROOT/'experiments/counterfactual_verification/comparison_mild_v2_browser.json'
    if browser.exists():
        text+=('\nFailure Case Explorer cho đúng bộ mild v2: [mở gallery](http://127.0.0.1:8789). '
            'Đã kiểm tra bằng trình duyệt: bộ lọc sigma0,5, ba dự đoán/case, đúng100 source gốc, '
            'background-only không gán accuracy và ảnh hiển thị khớp hash, không có page errors.\n\n'
            'Nếu cần mở lại:\n```powershell\n'
            '.\\.venv\\Scripts\\python.exe scripts\\failure_case_explorer.py --comparison-root outputs/mild_blur_v2 --port 8789\n```\n')
    (out/'report.md').write_text(text,encoding='utf-8')


def build_report(out):
    config=json.loads((out/'blur_config.json').read_text(encoding='utf-8'));previous=ROOT/config['previous_results']
    timings=json.loads((out/'timings.json').read_text())
    if sha256(out/'predictions.csv')!=timings['csv_sha256']:raise ValueError('New predictions changed after inference')
    if sha256(previous/'predictions.csv')!=config['previous_predictions_sha256']:raise ValueError('Previous predictions changed')
    frame=pd.read_csv(out/'predictions.csv');cohort=pd.read_csv(out/'cohort.csv');old=pd.read_csv(previous/'predictions.csv')
    for level,sigma in SIGMAS.items():
        selected=frame[(frame.control.isin(['natural','foreground_only']))&(frame.severity==level)]
        if len(selected)==0 or not (selected.sigma==sigma).all():raise ValueError('Actual sigma disagrees with v2 report protocol')
    if list(frame.columns)!=list(old.columns) or len(frame.columns)!=34:raise ValueError('Previous 34-column schema was not preserved')
    validation=validate_predictions(frame,cohort,severity_count=5)
    expected={'natural':6000,'foreground_only':1500,'background_only':1200,'original':300}
    if frame.control.value_counts().to_dict()!=expected:raise ValueError('Experimental condition counts disagree')
    missing=frame.isna().sum().to_dict()
    allowed={'true_y','is_correct'}
    if any(count and name not in allowed for name,count in missing.items()):raise ValueError('Unexpected missing values')
    if frame.true_y.isna().sum()!=1200 or frame.is_correct.isna().sum()!=1200:raise ValueError('Only background-only labels/correctness may be missing')
    shared=shared_comparison(frame,old);shared.to_csv(out/'shared_sigma_reproducibility.csv',index=False)
    summary,groups,classes=metric_tables(frame)
    for data in (summary,groups,classes):
        data['sigma']=data.severity.map({**SIGMAS,'original':0.})
        data['severity_role']=data.severity.map({**{l:'main' for l in MAIN},'extreme':'supplementary_diagnostic','original':'source_reference'})
    summary.to_csv(out/'metrics.csv',index=False);groups.to_csv(out/'group_metrics.csv',index=False);classes.to_csv(out/'class_metrics.csv',index=False)
    classes.to_csv(out/'prediction_bias.csv',index=False)
    natural=summary[summary.control=='natural'];drops=[]
    for r in natural.itertuples():
        c=natural[(natural.model_name==r.model_name)&(natural.severity=='clean')].iloc[0]
        ld,wd=c.landbird_accuracy-r.landbird_accuracy,c.waterbird_accuracy-r.waterbird_accuracy
        drops.append(dict(model_name=r.model_name,severity=r.severity,sigma=r.sigma,
            accuracy_drop=c.semantic_accuracy-r.semantic_accuracy,wga_drop=c.wga-r.wga,
            landbird_accuracy_drop=ld,waterbird_accuracy_drop=wd,class_asymmetric_drop=ld-wd,
            class_accuracy_gap=r.waterbird_accuracy-r.landbird_accuracy))
    drops=pd.DataFrame(drops);drops.to_csv(out/'degradation_metrics.csv',index=False)
    flips=paired_flips(frame);flips['sigma']=flips.severity.map(SIGMAS);flips.to_csv(out/'paired_flips.csv',index=False)
    flips.groupby(['model_name','severity','sigma','pair_type','true_y']).agg(
        source_foregrounds=('source_image_id','nunique'),paired_images=('flip','size'),flip_rate=('flip','mean'),
        land_to_water_rate=('land_to_water','mean'),water_to_land_rate=('water_to_land','mean'),
        mean_delta_p_waterbird=('delta_p_waterbird','mean')).reset_index().to_csv(out/'flip_metrics.csv',index=False)
    bg=background_table(frame,flips);bg.to_csv(out/'background_sensitivity.csv',index=False)
    ci=bootstrap_comparison(frame,samples=config['bootstrap_samples'],seed=config['sampling_seed'],levels=LEVELS,include_extra=True)
    ci['condition']='natural'
    ci=pd.concat([ci,neutral_bootstrap(frame,config['bootstrap_samples'],config['sampling_seed'])],ignore_index=True)
    ci['sigma']=ci.severity.map(SIGMAS);ci['severity_role']=ci.severity.map({**{l:'main' for l in MAIN},'extreme':'supplementary_diagnostic'})
    ci.to_csv(out/'bootstrap_ci.csv',index=False)
    validation.update(prediction_schema_columns=len(frame.columns),condition_rows=expected,missing_values=missing,
        exact_shared_previous_prediction_rows=int(shared.matched_rows.sum()),previous_cohort_hash_unchanged=sha256(out/'cohort.csv')==sha256(previous/'cohort.csv'),
        all_model_checkpoints_unchanged=True,main_sigmas=[0,.5,1,2],supplementary_sigmas=[4],human_recognizability='pending independent review')
    write_json(out/'validation.json',validation)
    figures(out,summary,classes,groups,flips)
    final_report(out,config,summary,classes,drops,ci,bg,validation,pd.read_csv(previous/'metrics.csv'))
    write_json(out/'analysis_provenance.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/counterfactual/failure_statistics.py']},
        new_inference=False,bootstrap_samples=config['bootstrap_samples'],numpy=np.__version__,pandas=pd.__version__))
    write_json(out/'status.json',dict(status='completed',prediction_rows=len(frame),input_records=3000,
        new_prediction_rows=timings['new_prediction_rows'],reused_prediction_rows=timings['reused_prediction_rows']))
    write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('integrity.json','files_manifest.csv')])
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(natural[['model_name','severity','sigma','semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy']].to_string(index=False))


if __name__=='__main__':build_report(ROOT/'outputs/mild_blur_v2')
