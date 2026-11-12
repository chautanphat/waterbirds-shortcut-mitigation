"""Plot saved DPE tables; no model computation. Requires matplotlib."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);a=parser.parse_args();root=a.results
read=lambda n:list(csv.DictReader((root/n).open()))
rows=read('per_seed.csv');parts=read('partition_sensitivity.csv')
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for family,color,label in [('mlp','#3569a8','Plain MLP'),('mlp_dropout','#c16c22','Dropout MLP')]:
 name=family+'-wd0.01';d=[float(next(r for r in rows if int(r['source_seed'])==s and r['configuration']==name)['delta_uniform_pp']) for s in range(5)]
 axes[0].plot(range(5),d,'o-',color=color,label=label,linewidth=1)
 gains=[np.mean(d)]
 for split in [1,2]:gains.append(np.mean([float(r['delta_uniform_within_partition_pp']) for r in parts if int(r['partition_seed'])==split and r['configuration']==name]))
 axes[1].plot(range(3),gains,'o-',color=color,label=label,linewidth=1)
axes[0].set(title='Primary partition: paired source differences',xlabel='Backbone/head source seed',ylabel='Test WGA change vs uniform (pp)',xticks=range(5))
axes[1].set(title='Same five sources: validation-partition sensitivity',xlabel='Validation partition seed',ylabel='Mean test WGA change vs uniform (pp)',xticks=range(3))
for ax in axes:ax.axhline(0,color='black',linewidth=.7,linestyle='--');ax.grid(alpha=.18);ax.legend(frameon=False)
fig.suptitle('Waterbirds / ResNet-50; weight decay 0.01; descriptive paired differences',fontsize=11)
for ext in ['svg','png']:fig.savefig(root/f'five_seed_findings.{ext}',dpi=180)
