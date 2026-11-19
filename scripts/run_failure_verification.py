"""Execute paired failure verification from cached real predictions; no training."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch

from src.counterfactual.common import read_csv,write_csv,write_json,sha256
from src.counterfactual.failure_statistics import validate_predictions,metric_tables
from src.counterfactual.failure_verification import (LEVELS,SIGMAS,MODELS,SCOPES,CONTRASTS,
    audit_images,paired_interventions,summarize_bootstrap,species_summaries)
from scripts.audit_failure_checkpoints import audit

NAMES={'POTER':'POTER','ERM_enhanced':'ERM enhanced','DPE_legacy':'DPE legacy'}
COLORS={'POTER':'#b64b39','ERM_enhanced':'#2878a4','DPE_legacy':'#437e4c'}


def figures(out,fg,stats,seed_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    xs=list(SIGMAS.values());primary=stats[(stats.bootstrap_unit.str.startswith('source_image_id'))&(stats.comparison=='within_seed')]
    def curve(ax,model,scope,metric,label=None,color=None):
        d=primary[(primary.model_name==model)&(primary.scope==scope)&(primary.metric==metric)].set_index('sigma').loc[xs]
        ax.plot(xs,100*d.estimate,'-o',label=label or NAMES[model],color=color or COLORS[model],lw=1.8,markersize=4)
        ax.fill_between(xs,100*d.ci95_low,100*d.ci95_high,color=color or COLORS[model],alpha=.10)
    def style(ax,title,ylabel):
        ax.set(title=title,xlabel='Gaussian sigma (model-view pixels)',ylabel=ylabel,xticks=xs)
        ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.15)
    def save(fig,name):
        fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
    for metric,filename,title in [('signed_probability_shift','background_probability_shift.png','Signed Waterbird probability shift'),
            ('flip_rate','paired_prediction_flips.png','Background-induced prediction flips')]:
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for y,scope in enumerate(['landbird','waterbird']):
            for j,contrast in enumerate(CONTRASTS):
                ax=axs[y,j]
                for model in MODELS:curve(ax,model,scope,contrast+'_'+metric)
                style(ax,scope.title()+': '+contrast.replace('_',' '),'Flip rate (%)' if metric=='flip_rate' else 'Signed probability shift (pp)')
                if metric=='flip_rate':ax.set_ylim(0,103)
                ax.legend(fontsize=8)
        fig.suptitle(title+' | foreground-clustered 95% CI, not training-seed uncertainty')
        save(fig,filename)
    fig,ax=plt.subplots(figsize=(8.7,5.2),layout='constrained')
    for model in MODELS:curve(ax,model,'all','interference')
    style(ax,'Natural-background accuracy minus neutral-gray accuracy','Accuracy change (pp)');ax.legend()
    save(fig,'background_interference_vs_sigma.png')
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained',sharey='row')
    for col,model in enumerate(MODELS):
        for row,metric in enumerate(['interference','amplification']):
            for scope,color in [('landbird','#b86e31'),('waterbird','#247fa3')]:curve(axs[row,col],model,scope,metric,scope.title(),color)
            style(axs[row,col],NAMES[model]+' / '+metric,'Accuracy change (pp)');axs[row,col].legend(fontsize=8)
    fig.suptitle('Negative interference: natural backgrounds reduce accuracy; negative amplification: extra reduction under blur')
    save(fig,'landbird_vs_waterbird_interference.png')
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    rng=np.random.default_rng(20261118)
    for col,model in enumerate(MODELS):
        for y in (0,1):
            ax=axs[y,col]
            d=fg[(fg.model_name==model)&(fg.true_y==y)]
            values=[100*d[d.sigma==s].delta_background.to_numpy() for s in xs]
            ax.boxplot(values,positions=xs,widths=.14,manage_ticks=False,showfliers=False)
            for sigma,v in zip(xs,values):ax.scatter(sigma+rng.uniform(-.045,.045,len(v)),v,s=11,alpha=.45,color=COLORS[model])
            style(ax,NAMES[model]+' / '+['Landbird','Waterbird'][y],'Per-foreground P(Water BG) − P(Land BG) (pp)')
    fig.suptitle('Each dot is one source foreground, averaged across two background pairs; panel scales differ')
    save(fig,'per_foreground_probability_shifts.png')
    # Missing seeds have no fabricated markers, means or zero-valued SD.
    for metric,filename,title,scope in [('wga','wga_across_seeds.png','Natural-composite WGA','all'),
        ('class_asymmetry','class_asymmetry_across_seeds.png','Class-asymmetric accuracy drop','all'),
        ('interference','background_interference_across_seeds.png','Landbird natural-minus-gray interference','landbird')]:
        fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
        for model,ax in zip(MODELS,axs):
            curve(ax,model,scope,metric,label=f"seed {1 if model=='POTER' else 0}")
            style(ax,NAMES[model]+' (1/3 checkpoints available)','WGA (%)' if metric=='wga' else 'Accuracy change (pp)')
            ax.legend(fontsize=8)
            if metric=='wga':ax.set_ylim(0,103)
        fig.suptitle(title+' | seed validation INCOMPLETE; bands describe foregrounds, training-seed SD is unavailable')
        save(fig,filename)
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained',sharey=True)
    for model,ax in zip(MODELS,axs):
        for scope,metric,color in [('Landbird','landbird_accuracy','#b86e31'),('Waterbird','waterbird_accuracy','#247fa3')]:
            d=seed_metrics[(seed_metrics.model_name==model)&(seed_metrics.control=='natural')].set_index('sigma').loc[xs]
            ax.plot(xs,100*d[metric],'-o',label=scope,color=color)
        style(ax,NAMES[model]+f" seed {1 if model=='POTER' else 0}",'Accuracy (%)');ax.set_ylim(0,103);ax.legend(fontsize=8)
    fig.suptitle('Individual available checkpoint only; no independent-seed replication has been established')
    save(fig,'class_accuracy_per_seed.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,metric,scope,title in zip(axs,['wga','class_asymmetry','interference'],['all','all','landbird'],
        ['WGA','Class asymmetry','Landbird background interference']):
        for i,model in enumerate(MODELS):
            d=primary[(primary.model_name==model)&(primary.sigma==2)&(primary.scope==scope)&(primary.metric==metric)].iloc[0]
            ax.errorbar(i,100*d.estimate,yerr=[[100*(d.estimate-d.ci95_low)],[100*(d.ci95_high-d.estimate)]],fmt='o',color=COLORS[model],capsize=4)
        ax.set(title=title,xticks=range(3),xticklabels=['POTER s1','ERM+ s0','DPE s0'],ylabel='WGA (%)' if metric=='wga' else 'Accuracy change (pp)')
        ax.axhline(0,color='gray',lw=.7)
    fig.suptitle('Sigma2 comparison | intervals resample foregrounds, not independently trained checkpoints')
    save(fig,'model_comparison_sigma2.png')


def mdtable(headers,rows):
    esc=lambda v:str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(esc,row))+' |' for row in rows])


def reports(out,config,fg,stats,seeds,across,manifest,validation):
    def find(model,metric,scope='landbird',sigma=2.,unit='source_image_id',comparison='within_seed'):
        return stats[(stats.model_name==model)&(stats.metric==metric)&(stats.scope==scope)&(stats.sigma==sigma)&
            (stats.comparison==comparison)&stats.bootstrap_unit.str.startswith(unit)].iloc[0]
    def effect(r):return f"{100*r.estimate:+.2f} [{100*r.ci95_low:+.2f}; {100*r.ci95_high:+.2f}]"
    hypotheses=[]
    for h,metric,scope,negative in [('H1','class_asymmetry','all',False),('H2','interference','landbird',True),('H3','amplification','landbird',True)]:
        r=find('POTER',metric,scope)
        supported=r.ci95_high< -1e-10 if negative else r.ci95_low>1e-10
        contrary=r.ci95_low>1e-10 if negative else r.ci95_high< -1e-10
        hypotheses.append(dict(hypothesis=h,status='Supported' if supported else 'Not supported' if contrary else 'Inconclusive',
            scope='available POTER seed1, sigma2, selected foreground/background cohort',effect=effect(r)+' pp',
            limitation='one checkpoint;4 fixed backgrounds; neutral scene statistics and composition artifacts; no independent human recognizability validation'))
    hypotheses += [dict(hypothesis='H4',status='Inconclusive',scope='independent POTER training seeds',effect='not estimable',
        limitation='Only seed1 available; seeds0/2 not trained; user requires notice/confirmation before training'),
        dict(hypothesis='H5',status='Inconclusive',scope='method-specific difference from ERM/DPE',effect='see paired contrasts; seed-level replication missing',
        limitation='Some checkpoint-conditioned background effects differ; class asymmetry is also present in DPE; unmatched training recipes/seeds')]
    write_csv(out/'hypothesis_evaluation.csv',hypotheses)
    point_rows=[]
    for model in MODELS:
        for scope in ['landbird','waterbird']:
            for sigma in SIGMAS.values():
                point_rows.append([NAMES[model],scope,sigma,effect(find(model,'interference',scope,sigma)),effect(find(model,'amplification',scope,sigma)),
                    effect(find(model,'delta_background',scope,sigma))])
    exp1=('# Experiment1: paired background intervention\n\n'
        '**Hoàn thành bằng dự đoán thật đã lưu, không inference lại.** '
        'Giữ nguyên source foreground, mask, placement, blur và preprocessing; so gray→Land A/B, gray→Water A/B và Land A/B→Water A/B. '
        'Mỗi contrast có2 background replicates; replicate được trung bình trong foreground trước bootstrap. '
        'Có7.200 paired rows từ6.000 prediction rows (100 foreground ×4 sigma ×5 nền ×3 model); chúng không phải7.200 mẫu độc lập.\n\n'
        'Primary sigma0/0,5/1/2; sigma4 bị loại khỏi phân tích chính. Các source tensors/ảnh render được kiểm lại. '
        'Landbird=0, Waterbird=1; unrounded probabilities. Logit margin là score(Water)−score(Land). '
        'DPE dùng log(mean head probability), nên không so độ lớn margin shift DPE với native CNN logits của POTER/ERM.\n\n'
        'Interference = accuracy natural−accuracy gray. Âm: natural backgrounds làm giảm accuracy. '
        'Amplification = interference(sigma)−interference(clean). Âm: phần giảm tăng thêm dưới blur. '
        'Delta_background = mean P(Water|water backgrounds)−mean P(Water|land backgrounds), có dấu.\n\n'
        +mdtable(['Model','Lớp','Sigma','Interference [CI95%] (pp)','Amplification [CI95%] (pp)','Delta background [CI95%] (pp)'],point_rows)+'\n\n'
        '### Nhạy cảm với species clustering\n\n'
        +mdtable(['Model','Sigma','Land interference: source bootstrap','Land interference: species bootstrap'],[
            [NAMES[m],s,effect(find(m,'interference','landbird',s)),effect(find(m,'interference','landbird',s,unit='species_id'))]
            for m in MODELS for s in [0.,2.]])+'\n\n'
        'Source bootstrap resample50 foreground/lớp. Species sensitivity resample10 species/lớp; giữ trung bình5 foreground/species. '
        'Cả hai có2.000 draws class-stratified, dùng cùng draws giữa model/sigma. Chúng không ước lượng training-seed uncertainty '
        'hay khả năng tổng quát sang background mới.\n\n'
        '### So sánh paired theo model tại sigma2\n\n'
        +mdtable(['So sánh','Land interference difference [CI95%] (pp)','Land amplification difference [CI95%] (pp)'],[
            ['POTER−'+NAMES[m],effect(find('POTER','interference',comparison='POTER_minus_'+m)),
             effect(find('POTER','amplification',comparison='POTER_minus_'+m))] for m in MODELS[1:]])+'\n\n'
        '### Phân bố và consistency\n\n')
    species=species_summaries(fg)
    for model in MODELS:
        d=species[(species.model_name==model)&(species.true_y==0)&(species.sigma==2)]
        raw=fg[(fg.model_name==model)&(fg.true_y==0)&(fg.sigma==2)]
        exp1+=f"- {NAMES[model]}: {int((d.interference<0).sum())}/10 loài Landbird có interference âm; {int((raw.interference<0).sum())}/50 foreground Landbird có interference âm tại sigma2.\n"
    exp1+=('\nĐổi nền là can thiệp trên input thực sự: foreground và blur được giữ cố định, nên chênh lệch đầu ra thuộc việc thay ảnh nền '
        'trong các cặp này. Nền có nhiều yếu tố ngoài thuộc tính Land/Water; gray cũng thay scene statistics. '
        'Kết quả chưa chỉ ra internal neural-network mechanism, learned class prior, hoặc nguyên nhân do riêng thuật toán POTER.\n\n')
    exp1+=('### Các contrast của POTER tại sigma2\n\n'
        +mdtable(['Lớp','Contrast','Flip rate [CI95%] (%)','Signed ΔP(Water) [CI95%] (pp)',
            'Mean absolute ΔP [CI95%] (pp)','Margin shift [CI95%] (logit units)','Accuracy change [CI95%] (pp)'],[
            [scope,contrast,effect(find('POTER',contrast+'_flip_rate',scope)),
             effect(find('POTER',contrast+'_signed_probability_shift',scope)),
             effect(find('POTER',contrast+'_absolute_probability_shift',scope)),
             f"{find('POTER',contrast+'_margin_shift',scope).estimate:+.4f} [{find('POTER',contrast+'_margin_shift',scope).ci95_low:+.4f}; {find('POTER',contrast+'_margin_shift',scope).ci95_high:+.4f}]",
             effect(find('POTER',contrast+'_accuracy_change',scope))]
            for scope in ['landbird','waterbird'] for contrast in CONTRASTS])+'\n\n'
        '**Không đồng nhất background interference với ưu tiên nền Water.** '
        f"Với Landbird sigma2, Land→Water làm ΔP(Water) {effect(find('POTER','delta_background'))} pp. "
        'Nền Water làm giảm P(Water) so với nền Land trong các cặp đã chọn. '
        'Cả Land và Water vẫn có thể làm tăng P(Water) so với gray; độ lớn và hướng phụ thuộc contrast. '
        'Điều này không hỗ trợ câu chuyện đơn giản water-background→Waterbird prediction.\n\n'
        '### Trả lời câu hỏi Experiment1\n\n'
        '1. Có. Đổi nền giữ foreground cố định làm thay đổi output và correctness; xem flip rates/CI ở trên.\n'
        '2. Landbird interference tăng thêm dưới blur tại sigma2 so với clean; amplification CI nằm dưới0. '
        'Mọi sigma đều được báo cáo, không chỉ endpoint.\n'
        '3. Chênh Landbird−Waterbird interference tại sigma2 là '
        +effect(find('POTER','land_minus_water_interference','all'))+' pp.\n'
        '4. Các checkpoint ERM/DPE có response khác POTER trên cùng inputs; training-method specificity chưa xác nhận.\n'
        '5. Hiệu ứng âm xuất hiện ở9/10 loài Landbird của POTER,19/50 foreground. Species sensitivity CI '
        'vẫn âm nhưng independent training-seed consistency chưa được kiểm tra.\n\n')
    for name in ['background_probability_shift.png','paired_prediction_flips.png','background_interference_vs_sigma.png',
        'landbird_vs_waterbird_interference.png','per_foreground_probability_shifts.png']:
        exp1+=f'![{name}]({name})\n\n'
    (out/'exp1_summary.md').write_text(exp1,encoding='utf-8')
    htable=mdtable(['Hypothesis','Trạng thái','Phạm vi','Effect / CI95%'],[[h['hypothesis'],h['status'],h['scope'],h['effect']] for h in hypotheses])
    perseed=seeds[seeds.control=='natural']
    text=('# POTER Failure Verification: Experiments1 &2\n\n'
        '## 1. Executive summary\n\n'
        '**Experiment1 đã hoàn thành; Experiment2 chưa hoàn thành xác nhận đa seed.** '
        f"POTER seed1, Landbird tại sigma2: natural−gray interference **{effect(find('POTER','interference'))} pp**; "
        f"amplification so với clean **{effect(find('POTER','amplification'))} pp**. "
        'Đây là bằng chứng can thiệp input có điều kiện trên checkpoint và bốn nền đã chọn. '
        'Chưa xác nhận tính ổn định giữa các seed huấn luyện độc lập.\n\n'
        f"Land→Water tại sigma2 cho Landbird có signed ΔP(Water) **{effect(find('POTER','delta_background'))} pp**. "
        'Nền Water trong ngân hàng này làm giảm P(Water) so với Land; không suy ra cơ chế water-background→Waterbird bias.\n\n'
        '**Quyết định: NO-GO / INCONCLUSIVE** cho chuyển sang Mechanism Discovery ở cấp phương pháp hoặc phát triển mitigation. '
        'Cần xác nhận đa checkpoint và rà soát artifacts/recognizability trước. Không triển khai phương pháp robustness mới.\n\n'
        'Không có training hoặc inference mới trong verification này. Experiment2 tái sử dụng6.300 rows của ba checkpoint hiện có '
        '(2.100/model:1.600 natural,400 gray,100 original), so với mục tiêu18.900. Thiếu12.600 rows từ6 checkpoint độc lập còn thiếu. '
        'Yêu cầu mới nhất của người dùng: báo trước và chờ xác nhận nếu cần training, người dùng không thể train local.\n\n'
        '## 2. Experiment1 results\n\n'
        +mdtable(['Model','Lớp','Sigma','Interference [CI95%] (pp)','Amplification [CI95%] (pp)'],[
            [NAMES[m],scope,s,effect(find(m,'interference',scope,s)),effect(find(m,'amplification',scope,s))]
            for m in MODELS for scope in ['landbird','waterbird'] for s in [0.,2.]])+'\n\n'
        'Các background replicates được giữ riêng trong exp1_paired_interventions.csv, sau đó trung bình trong source foreground '
        'trước CI. exp1_foreground_effects.csv giữ effect từng foreground; exp1_species_summary.csv giữ từng species. '
        'Paired flip, signed/absolute probability shift và margin shift được tính cho cả gray→Land, gray→Water, Land→Water. '
        'Xem exp1_summary.md và statistical_analysis.csv để đọc mọi sigma/contrast/class, bao gồm source và species sensitivity bootstrap.\n\n'
        'Tại sigma2/Landbird, POTER có gray→Land flip rate '
        +effect(find('POTER','gray_to_land_flip_rate'))+'%, gray→Water '
        +effect(find('POTER','gray_to_water_flip_rate'))+'%, Land→Water '
        +effect(find('POTER','land_to_water_flip_rate'))+'%. '
        'Điều kiện Land có Landbird accuracy58%, Water72%, gray86%. Water thay vì Land giảm P(Water) '
        'và tăng correctness trong các cặp này, dù cả hai nhóm natural backgrounds có accuracy thấp hơn gray. '
        'Hướng effect phụ thuộc contrast; không gọi đây là một semantic water-cue mechanism đã xác nhận.\n\n'
        '![Interference](landbird_vs_waterbird_interference.png)\n\n'
        '## 3. Experiment2 results\n\n'
        +mdtable(['Model','Seed có sẵn','Seed còn thiếu','Trạng thái'],[
            [NAMES[m],1 if m=='POTER' else 0,', '.join(map(str,manifest['missing_seeds'][m])),'1/3; chưa xác nhận đa seed'] for m in MODELS])+'\n\n'
        'Có hai image backbones duy nhất: POTER và enhanced ERM; DPE dùng chung backbone ERM. '
        'Trong từng method chỉ có một training seed hợp lệ. Bản sao byte-identical không tăng n; '
        'các head-fitting seed/fold không thay thế image-backbone training seeds. '
        'last_training.pt là snapshot cùng POTER seed1; checkpoint bs16/smoke cũng seed1 và khác recipe. '
        'Log POTER seed2/3 trên D chỉ dừng ở epoch0, chưa có checkpoint; nguyên nhân lỗi không được ghi trong log.\n\n'
        'Downloads có thêm standard-ERM seed3 và DPE partition0 tương ứng, với hash khớp archived lineage. '
        'Chúng khác enhanced-ERM400 và DPE legacy/full-validation recipe, nên không dùng để giả hoàn thành seed target. '
        'Manifest còn ghi các recovery-test artifacts và những head-only files đã loại. '
        'Một số manifest tham chiếu checkpoint ở đường dẫn /kaggle/working nhưng file không có trong checkout; '
        'đường dẫn metadata không được tính như một checkpoint đã có/evaluate.\n\n'
        '### Từng checkpoint hiện có\n\n'
        +mdtable(['Model','Seed','Sigma','AA (%)','WGA (%)','Landbird (%)','Waterbird (%)','Asymmetry (pp)'],[
            [NAMES[r.model_name],r.model_seed,r.sigma,*[f'{100*v:.2f}' for v in [r.semantic_accuracy,r.wga,r.landbird_accuracy,r.waterbird_accuracy,r.class_asymmetry]]]
            for r in perseed.itertuples()])+'\n\n'
        'Mean với n=1 bằng đúng giá trị của checkpoint; **SD giữa training seeds không xác định**, không ghi0. '
        'exp2_across_seed_summary.csv đánh dấu thiếu6 checkpoint. Foreground-bootstrap CIs trong statistical_analysis.csv '
        'không phải training-seed CI. Các plot across_seeds hiển thị một seed/method và ghi rõ1/3, không vẽ seed giả.\n\n'
        '### Training chưa khởi chạy\n\n'
        f"Run POTER gốc ghi peak VRAM **{manifest['training']['historical_poter_peak_vram_gib']:.2f} GiB** "
        '(batch128, FP32), vượt GPU local6 GiB. Không đổi batch, AMP, BN hoặc loss để ép train. '
        f"Log300 epochs có trung bình {manifest['training']['historical_poter_mean_epoch_seconds']:.2f}s/epoch; "
        f"hai POTER seeds bổ sung ước tính {manifest['training']['two_additional_poter_seeds_serial_hours']:.2f} giờ nối tiếp "
        'trên GPU tương đương T4. Đây không phải benchmark mới hoặc bảo đảm thời gian cloud. ERM/DPE sẽ cần thêm thời gian. '
        'Train4795/val1199 images có đủ ởD và metadata canonical khớp; dữ liệu không phải blocker. '
        'Training chưa được bắt đầu theo chỉ dẫn báo trước của người dùng; cần cloud đủ VRAM và xác nhận trước khi launch. '
        'pending_training_plan.json khóa các yêu cầu seed/recipe để xem xét; file này không thực thi training.\n\n'
        '![WGA seeds](wga_across_seeds.png)\n\n![Asymmetry seeds](class_asymmetry_across_seeds.png)\n\n'
        '![Interference seeds](background_interference_across_seeds.png)\n\n![Class seed](class_accuracy_per_seed.png)\n\n'
        '## 4. Model comparison\n\n'
        +mdtable(['POTER−baseline tại sigma2','Land interference difference [CI95%] (pp)','Asymmetry difference [CI95%] (pp)'],[
            ['POTER−'+NAMES[m],effect(find('POTER','interference',comparison='POTER_minus_'+m)),
             effect(find('POTER','class_asymmetry',scope='all',comparison='POTER_minus_'+m))] for m in MODELS[1:]])+'\n\n'
        'Tách clean performance, drop do blur và effect background. POTER/DPE có class-asymmetric blur response; '
        'enhanced ERM có WGA thấp hơn ởsigma2. Phản ứng natural−gray khác nhau ở checkpoint hiện có, '
        'nhưng seed/recipe không khớp và DPE legacy thiếu historical backbone binding. '
        'Không quy khác biệt quan sát được cho thuật toán POTER. Xác suất chưa calibration chung; '
        'DPE log-mean-probability margin không cùng thang native CNN logits.\n\n'
        '![Sigma2](model_comparison_sigma2.png)\n\n'
        '## 5. Hypothesis evaluation\n\n'+htable+'\n\n'
        'H1: Landbird degradation lớn hơn Waterbird. H2: natural backgrounds làm giảm Landbird accuracy so với gray khi blur. '
        'H3: interference tăng thêm dưới blur. H4: tái hiện giữa POTER training seeds độc lập. '
        'H5: POTER failure behavior khác ERM/DPE ở cấp phương pháp. Supported choH1–H3 nếu có chỉ áp dụng '
        'POTER seed1 và input cohort hiện tại; không suy sangH4/H5. Không hiệu chỉnh multiple comparisons.\n\n'
        '## 6. Mechanism Discovery decision\n\n'
        '**NO-GO / INCONCLUSIVE.** Các can thiệp input cung cấp giả thuyết background interference đáng kiểm tra tiếp. '
        'Thiếu independent POTER seeds ngăn xác nhận tính ổn định ở cấp method. Gray thay đổi scene statistics; '
        'bốn nền có người/động vật/objects và một số mask/crop có artifacts. Human recognizability chưa được đánh giá độc lập. '
        'Không chọn GO cho internal mechanism chỉ từ các output shifts.\n\n'
        'Bằng chứng tối thiểu còn thiếu: bổ sung đúng2 training seeds/method với fixed recipe và validation rule, '
        'gắn backbone/head hashes và lineage DPE; đánh giá2.100 inputs/checkpoint đã khóa; báo individual seeds và mean±SD '
        'thật; kiểm tra mask/recognizability độc lập và backgrounds/cohort mới để xác nhận sau nghiên cứu exploratory. '
        'Ba seeds chỉ là bằng chứng training variation sơ bộ. Không dùng cohort test đã xem để phát triển/tune mitigation.\n\n'
        '## 7. Research contribution assessment\n\n'
        '- Tái lập được cặp input/hash và tính effect trong một checkpoint; chưa xác nhận lặp giữa training seeds.\n'
        '- POTER-specificity chưa được hỗ trợ ở cấp phương pháp; DPE cũng có class-asymmetric degradation.\n'
        '- Bằng chứng vượt một accuracy-drop plot đơn thuần: cùng foreground/blur/mask/crop được can thiệp nền, '
        'có gray reference, replicate-level pairs, interference/amplification, distributions và species sensitivity.\n'
        '- Hướng ưu tiên là xác minh background-induced interference bằng checkpoints mới trước khi phân tích representation/logit shifts.\n'
        '- Chưa đủ để phát triển mitigation hoặc tuyên bố novelty/paper-readiness. Học class priors, calibration '
        'và internal representation geometry vẫn là giả thuyết chưa được xác định.\n\n'
        '### Integrity, execution and files\n\n'
        'Nguồn được gọi predictions(5).csv trong handoff không được đính kèm; dùng canonical '
        'outputs/mild_blur_v2/predictions.csv9.000 rows,34 columns, hash khóa trong experiment_config.json. '
        'Không thay sigma/cohort/checkpoint hoặc sửa kết quả cũ. '
        'statistical_analysis.csv ghi2.000 class-stratified bootstrap draws theo source foreground và sensitivity theo species. '
        'Mọi background/severity của cùng cluster đi cùng nhau; model comparisons paired cùng input. '
        'Không pool composite hay seed như quan sát độc lập.\n\n'
        'Required outputs đã có: exp1_paired_interventions.csv, exp1_summary.md, exp2_seed_metrics.csv, '
        'exp2_group_metrics.csv, exp2_class_metrics.csv, exp2_predictions.csv, statistical_analysis.csv, '
        'checkpoint_manifest.json, experiment_config.json, final_report.md và10 requested PNG plots. '
        'Các file exp2 chứa đúng available seeds, không chứa rows giả cho seeds thiếu. '
        'files_manifest.csv liệt kê toàn bộ file/bytes/SHA256. validation.json ghi coverage vàimage audits; '
        'checkpoint_candidates.csv ghi lý do chấp nhận/loại từng asset.\n\n'
        'Chạy lại analysis:\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_failure_verification.py\n```\n'
        'Đổi output_root trong config để giữ một version mới. Script không train hoặc chạy model inference.\n')
    if (out/'tests.log').exists():text+='\nAutomated checks: '+(out/'tests.log').read_text(encoding='utf-8').strip().splitlines()[-1]+'\n'
    (out/'final_report.md').write_text(text,encoding='utf-8')
    write_json(out/'decision.json',dict(decision='NO-GO / INCONCLUSIVE',experiment1='completed',experiment2='incomplete',
        independent_seed_replication_confirmed=False,training_started=False,new_inference_rows=0,
        next_action='Notify and obtain user confirmation for cloud training of missing independently trained seeds'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','report'],default='all')
    args=parser.parse_args()
    start=time.perf_counter();config=json.loads((ROOT/'configs/failure_verification.json').read_text(encoding='utf-8'))
    if config['allow_training'] or config['allow_new_inference']:raise ValueError('This verification entrypoint is cache-only and never trains')
    out=ROOT/config['output_root'];out.mkdir(exist_ok=True,parents=True);source=ROOT/config['source_root']
    if args.stage=='report':
        locked=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
        saved=json.loads((out/'integrity.json').read_text(encoding='utf-8'))['sha256']
        required=['exp1_foreground_effects.csv','statistical_analysis.csv','exp2_seed_metrics.csv',
            'exp2_across_seed_summary.csv','checkpoint_manifest.json','validation.json','exp2_predictions.csv']
        for name in required:
            if sha256(out/name)!=saved[name]:raise ValueError('Saved analysis changed: '+name)
        if sha256(source/'predictions.csv')!=locked['source_predictions_sha256']:raise ValueError('Source predictions changed')
        load=lambda name:pd.read_csv(out/name,float_precision='round_trip')
        manifest=json.loads((out/'checkpoint_manifest.json').read_text(encoding='utf-8'))
        validation=json.loads((out/'validation.json').read_text())
        foregrounds=load('exp1_foreground_effects.csv');statistics=load('statistical_analysis.csv')
        seeds=load('exp2_seed_metrics.csv')
        figures(out,foregrounds,statistics,seeds)
        reports(out,locked,foregrounds,statistics,seeds,load('exp2_across_seed_summary.csv'),manifest,validation)
        provenance=json.loads((out/'analysis_provenance.json').read_text())
        provenance.setdefault('report_refreshes',[]).append(dict(utc=datetime.now(timezone.utc).isoformat(),
            reporter_sha256=sha256(Path(__file__)),new_training=False,new_inference=False,statistical_csvs_unchanged=True))
        write_json(out/'analysis_provenance.json',provenance)
        write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('integrity.json','files_manifest.csv')])
        write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
        print('Report refreshed; all saved statistical inputs verified, no inference/training executed.');return
    torch.set_num_threads(4)
    integrity=json.loads((source/'integrity.json').read_text(encoding='utf-8'))['sha256']
    for name,digest in integrity.items():
        if sha256(source/name)!=digest:raise ValueError('Source result changed: '+name)
    frame=pd.read_csv(source/'predictions.csv',float_precision='round_trip');cohort=pd.read_csv(source/'cohort.csv')
    validate_predictions(frame,cohort,severity_count=5)
    if set(frame.model_seed[frame.model_name=='POTER'])!={1} or any(set(frame.model_seed[frame.model_name==m])!={0} for m in MODELS[1:]):
        raise ValueError('Unexpected checkpoint seed identifiers')
    config.update(started_utc=datetime.now(timezone.utc).isoformat(),source_predictions_sha256=sha256(source/'predictions.csv'),
        source_integrity_sha256=sha256(source/'integrity.json'),source_cohort_sha256=sha256(source/'cohort.csv'),
        requested_filename='predictions(5).csv',requested_file_supplied=False,
        input_resolution='Canonical audited9000-row saved predictions, not an assumed browser-renamed copy')
    write_json(out/'experiment_config.json',config)
    manifest=audit(config,out)
    print('Checkpoint audit complete; missing seeds remain untrained.',flush=True)
    source_config=json.loads((source/'blur_config.json').read_text(encoding='utf-8'))
    image_audit=audit_images(frame,read_csv(source/'cohort.csv'),source_config,ROOT/source_config['data_root'],out)
    print('Primary/control images and tensors re-rendered and verified; no model inference.',flush=True)
    pairs,fg=paired_interventions(frame)
    pairs.to_csv(out/'exp1_paired_interventions.csv',index=False);fg.to_csv(out/'exp1_foreground_effects.csv',index=False)
    species=species_summaries(fg);species.to_csv(out/'exp1_species_summary.csv',index=False)
    stats=summarize_bootstrap(fg,samples=config['bootstrap_samples'],seed=config['sampling_seed'])
    if config['additional_species_bootstrap']:
        stats=pd.concat([stats,summarize_bootstrap(species,samples=config['bootstrap_samples'],seed=config['sampling_seed'],unit='species_id')],ignore_index=True)
    stats.to_csv(out/'statistical_analysis.csv',index=False)
    # Preserve original CSV fields and decimal strings for all reused prediction rows.
    selected=[r for r in read_csv(source/'predictions.csv') if r['control']=='original' or
        r['control'] in ('natural','foreground_only') and r['severity'] in LEVELS]
    if len(selected)!=6300:raise ValueError('Expected2100 cached predictions per available method checkpoint')
    write_csv(out/'exp2_predictions.csv',selected)
    seed_frame=frame[(frame.control=='original')|((frame.control.isin(['natural','foreground_only']))&frame.severity.isin(LEVELS))].copy()
    seeds,groups,classes=metric_tables(seed_frame)
    for data in (seeds,groups,classes):
        data['model_seed']=data.model_name.map({'POTER':1,'ERM_enhanced':0,'DPE_legacy':0})
        data['sigma']=data.severity.map({**SIGMAS,'original':0.})
    drops=[]
    for r in seeds.itertuples():
        if r.control=='original':drops.append(dict(wga_drop=None,landbird_drop=None,waterbird_drop=None,class_asymmetry=None));continue
        c=seeds[(seeds.model_name==r.model_name)&(seeds.control==r.control)&(seeds.severity=='clean')].iloc[0]
        ld,wd=c.landbird_accuracy-r.landbird_accuracy,c.waterbird_accuracy-r.waterbird_accuracy
        drops.append(dict(wga_drop=c.wga-r.wga if r.control=='natural' else None,landbird_drop=ld,waterbird_drop=wd,class_asymmetry=ld-wd))
    seeds=pd.concat([seeds.reset_index(drop=True),pd.DataFrame(drops)],axis=1)
    seeds.to_csv(out/'exp2_seed_metrics.csv',index=False);groups.to_csv(out/'exp2_group_metrics.csv',index=False);classes.to_csv(out/'exp2_class_metrics.csv',index=False)
    across=[]
    for r in seeds[seeds.control=='natural'].itertuples():
        for metric in ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy','wga_drop','class_asymmetry']:
            across.append(dict(model_name=r.model_name,sigma=r.sigma,metric=metric,available_seeds=1,required_seeds=3,
                mean=getattr(r,metric),training_seed_std=None,status='incomplete; SD undefined with1 independent seed'))
    across=pd.DataFrame(across);across.to_csv(out/'exp2_across_seed_summary.csv',index=False)
    validation=dict(experiment1_pair_rows=len(pairs),foreground_effect_rows=len(fg),source_foregrounds=fg.source_image_id.nunique(),
        species=fg.species_id.nunique(),source_prediction_rows=len(frame),experiment2_prediction_rows=len(selected),
        target_exp2_prediction_rows=18900,missing_exp2_prediction_rows=12600,new_inference_rows=0,new_training_checkpoints=0,
        reused_predictions=len(selected),image_audit=image_audit,bootstrap_samples=config['bootstrap_samples'],
        duplicates=int(pairs.duplicated(['model_name','model_seed','source_image_id','sigma','contrast','replicate']).sum()),
        seed_validation='incomplete',training_policy=config['training_policy'])
    if validation['duplicates']:raise ValueError('Duplicate paired intervention records')
    write_json(out/'validation.json',validation)
    figures(out,fg,stats,seeds);reports(out,config,fg,stats,seeds,across,manifest,validation)
    write_json(out/'status.json',dict(experiment1='completed',experiment2='incomplete_missing_independent_seeds',
        decision='NO-GO / INCONCLUSIVE',new_training=0,new_inference=0,elapsed_seconds=time.perf_counter()-start))
    write_json(out/'analysis_provenance.json',dict(source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [
        Path(__file__),ROOT/'scripts/audit_failure_checkpoints.py',ROOT/'src/counterfactual/failure_verification.py']},
        python=sys.version,numpy=np.__version__,pandas=pd.__version__,torch=str(torch.__version__),
        analysis_seed=config['sampling_seed'],checkpoint_selection='fixed existing checkpoints; no test-driven selection'))
    write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('integrity.json','files_manifest.csv')])
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(validation),flush=True)


if __name__=='__main__':main()
