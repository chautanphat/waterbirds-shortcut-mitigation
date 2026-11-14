"""Development score/reliability plots with distinct-image support counts."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.discrete_weighting.store import atomic
from .journal import csvwrite


def score_cells(p,scores,y,ids):
    pred=p.argmax(-1);correct=pred==y[:,None];rows=[]
    for k in (0,1):
        mask=pred==k;values=np.unique(scores[mask])
        for value in values:
            m=mask&(scores==value);n=int(m.sum())
            rows.append(dict(predicted_class=k,score=float(value),observations=n,distinct_images=int(m.any(1).sum()),
                             empirical_correctness=float(correct[m].mean()),low_support=int(m.any(1).sum())<30))
    return rows


def phase1_figures(root,winner):
    root=Path(root);paths=[]
    for seed in range(3):
        mid=winner['map_id'];lookup=json.loads((root/f'phase1fit/{mid}/{seed}/lookup.json').read_text());cells=pd.DataFrame(lookup['cells'])
        name=f"grid_{mid}_a{winner['alpha']:g}_b{winner['beta']:g}_s{seed}"
        held=pd.read_csv(root/'phase1/candidates'/name/'heldout_score_correctness.csv')
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for k in (0,1):
            t=cells[cells.predicted_class==k]
            for ax,key in zip(axes[k,:2],['score','empirical_correctness']):
                m=t.pivot(index='confidence_bin',columns='agreement_bin',values=key);n=t.pivot(index='confidence_bin',columns='agreement_bin',values='distinct_images')
                ax.imshow(m,aspect='auto');ax.set(title=f'class {k}: train {key}',xlabel='LOO agreement bin',ylabel='Confidence quantile bin');ax.set_xticks(range(3));ax.set_yticks(range(3))
                for i in range(3):
                    for j in range(3):
                        v=m.iloc[i,j];count=n.iloc[i,j]
                        ax.text(j,i,f'{v:.3f}\nn={count}'+('*' if count<30 else ''),ha='center',color='white',fontsize=8)
            ax=axes[k,2];z=held[held.predicted_class==k];ax.scatter(z.score,z.empirical_correctness,s=np.maximum(z.distinct_images/3,5))
            for r in z.itertuples():ax.annotate(f'n={r.distinct_images}',(r.score,r.empirical_correctness),fontsize=6)
            ax.set(title=f'class {k}: meta_select (post-lock)',xlabel='Inference lookup score',ylabel='Expert correctness')
        fig.suptitle(f'EXPLORATORY DEVELOPMENT | {mid} seed {seed} | counts = distinct images; * <30')
        fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
        p=root/f'phase1/plots/primary_seed{seed}.png';atomic(p,buffer.getvalue());paths.append(p)
        p=root/f'phase1/plots/primary_seed{seed}_source.csv';csvwrite(p,[dict(view='train_cell',**r) for r in lookup['cells']]+[dict(view='select_score',**r) for r in held.to_dict('records')]);paths.append(p)
    return paths
