import json,collections
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/root/autodl-tmp/Glasseek/analysis/released-model-inference-url');data=json.loads((P/'results.json').read_text());cs=['#5278bf','#d39a38','#53a397'];names=list(data);totals={}
fig,axes=plt.subplots(1,2,figsize=(14,5.8),gridspec_kw={'width_ratios':[1.1,1.25]});fig.suptitle('URL token share in locally recorded model rollouts',x=.06,ha='left',weight='bold',fontsize=17);fig.text(.06,.90,'Qwen3.5-9B tokenizer | all available cases in each selected run | no system / chat-template tokens',fontsize=9.5,color='#555f6a')
for i,(name,d) in enumerate(data.items()):
 rows=d['cases'];s=collections.Counter()
 for r in rows:s.update({k:v for k,v in r.items() if type(v) is int})
 model=100*s['model_url']/s['model_tokens'];tool=100*s['tool_url']/s['tool_tokens'];allpct=100*(s['model_url']+s['tool_url']+s['question_url'])/(s['model_tokens']+s['tool_tokens']+s['question_tokens']);totals[name]={'cases':len(rows),'generated':{'tokens':s['model_tokens'],'url_tokens':s['model_url'],'percent':model},'tool':{'tokens':s['tool_tokens'],'url_tokens':s['tool_url'],'percent':tool},'visible_trajectory':{'tokens':s['model_tokens']+s['tool_tokens']+s['question_tokens'],'url_tokens':s['model_url']+s['tool_url']+s['question_url'],'percent':allpct},'compaction_excluded_steps':s['compaction_excluded_steps'],'final_answer_cases':sum(bool(r['has_final']) for r in rows)}
 pos=np.arange(3)+i*.24;axes[0].barh(pos,[model,tool,allpct],height=.23,label=name,color=cs[i]);
 for yy,v in zip(pos,[model,tool,allpct]):axes[0].text(v+.35,yy,f'{v:.2f}%',va='center',fontsize=8)
 a=np.cumsum([r.get('model_url',0) for r in rows]);b=np.cumsum([r.get('model_tokens',0) for r in rows]);axes[1].plot(np.arange(1,len(rows)+1),100*a/np.maximum(b,1),label=name,color=cs[i],linewidth=2)
axes[0].set_yticks(np.arange(3)+.24,['Model output','Tool results','Visible trajectory']);axes[0].invert_yaxis();axes[0].set_xlabel('URL token share (%)');axes[0].set_xlim(0,max(v['tool']['percent'] for v in totals.values())*1.22);axes[0].set_title('Cumulative share across each run',loc='left',weight='bold');axes[1].set_title('Model-output cumulative share',loc='left',weight='bold');axes[1].set_xlabel('Cases included, viewer order');axes[1].set_ylabel('URL share in model output (%)')
for a in axes:a.grid(axis='x',alpha=.16);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.17),frameon=False,fontsize=8)
fig.text(.06,.02,'Viewer-normalized traces: tool-call displays are reconstructed; compaction-marked steps excluded from model output.\nQUEST and Glasseek use different benchmarks and setups. BrowseComp runs include unfinished trajectories.',fontsize=9,color='#505a66');fig.subplots_adjust(left=.13,right=.97,top=.79,bottom=.26,wspace=.32);fig.savefig(P/'inference-url-share.png',dpi=180,bbox_inches='tight');fig.savefig(P/'inference-url-share.pdf',bbox_inches='tight');(P/'summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2));print(json.dumps(totals,ensure_ascii=False,indent=2))
