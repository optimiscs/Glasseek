import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;r=json.loads((p/'report.json').read_text());s=r['summary'];modes=list(s)
colors=['#7b8494','#168b83','#3776c3','#9a66c8']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
for ax,(title,key,scale,unit) in zip(axes.flat,[('Reference-answer accuracy','accuracy',100,'%'),('Cumulative input incl. cache','input_tokens_including_cache',1e-6,'M tokens'),('Generated output','output_tokens',1e-3,'K tokens'),('Observed group completion time','wall_seconds',1/3600,'hours')]):
 values=[s[m][key]*scale for m in modes];bars=ax.bar(modes,values,color=colors,width=.58)
 for bar,m,v in zip(bars,modes,values):
  label=f'{v:.1f}'
  if key=='accuracy':label+=f'% ({s[m]["correct"]}/34)'
  ax.text(bar.get_x()+bar.get_width()/2,v,label,ha='center',va='bottom',fontsize=10)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('URL management: 34 questions per mode, one live run',fontsize=17,fontweight='bold')
fig.supxlabel('Shared GPU, changing web results, retrieval outage, and code hash drift limit causal comparisons.',fontsize=9)
fig.savefig(p/'comparison.png',dpi=170);fig.savefig(p/'comparison.svg')
