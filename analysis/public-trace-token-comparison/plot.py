import json,collections
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/root/autodl-tmp/Glasseek/analysis/public-trace-token-comparison');data=json.loads((P/'results.json').read_text())['data'];names=list(data); colors=['#5374b7','#5eafa9','#e2a445','#b7becb']; roles=['assistant','tool_result','user','other'];cats=['url','number','markup','other']; catlabels=['URL','Numbers','Tags / brackets','Other text'];rng=np.random.default_rng(20261105)
summary={};x=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1.15,1.35,1]});fig.patch.set_facecolor('#ffffff')
for j,(n,d) in enumerate(data.items()):
 rows=d['rows'];role=collections.Counter();cat=collections.Counter();u=collections.Counter()
 for r in rows:role.update(r['roles']);cat.update(r['categories']);u.update(r['url_by_role'])
 denom=sum(cat.values());assert denom==sum(v for k,v in role.items() if k!='system'),(n,denom,role)
 rolep={k:100*role[k]/denom for k in roles};catp={k:100*cat[k]/denom for k in cats}
 sampled=rng.integers(0,len(rows),size=(1200,len(rows)));a=np.array([r['categories'].get('url',0) for r in rows]);b=np.array([sum(r['categories'].values()) for r in rows]);boot=(a[sampled].sum(axis=1)/b[sampled].sum(axis=1))*100;ci=np.percentile(boot,[2.5,97.5]);summary[n]={'population':d['population'],'sample':len(rows),'content_tokens':denom,'system_tokens':role['system'],'roles':dict(role),'url_by_role':dict(u),'category_counts':dict(cat),'category_percent':catp,'url_95pct_bootstrap_ci':ci.tolist(),'role_percent_excluding_system':rolep}
 bottom=0
 for i,k in enumerate(roles):
  v=rolep[k];axes[0].barh(j,v,left=bottom,color=colors[i],height=.56);bottom+=v
 bottom=0
 for i,k in enumerate(cats):
  v=catp[k];axes[1].barh(j,v,left=bottom,color=colors[i],height=.56);bottom+=v
 v=catp['url'];axes[2].barh(j,v,color=colors[0],height=.45);axes[2].errorbar(v,j,xerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color='#283447',capsize=3,fmt='none');axes[2].text(max(v,ci[1])+0.04,j,f'{v:.2f}%',va='center',fontsize=11)
for a in axes:
 a.set_yticks(x,names);a.invert_yaxis();a.grid(axis='x',alpha=.18);a.set_axisbelow(True);a.spines[['top','right','left']].set_visible(False);a.tick_params(axis='y',length=0)
axes[0].set_title('Who produced the tokens?',loc='left',weight='bold');axes[0].set_xlabel('Share of non-system tokens (%)');axes[0].set_xlim(0,100);axes[0].legend(roles,loc='lower center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=8)
axes[1].set_title('Content category',loc='left',weight='bold');axes[1].set_xlabel('Share of non-system tokens (%)');axes[1].set_xlim(0,100);axes[1].legend(catlabels,loc='lower center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=8)
axes[2].set_title('URL share, enlarged',loc='left',weight='bold');axes[2].set_xlabel('URL tokens / non-system tokens (%)');axes[2].set_xlim(0,max(summary[n]['url_95pct_bootstrap_ci'][1] for n in names)*1.4+.2)
for a in axes[1:]:a.tick_params(labelleft=False)
fig.suptitle('Public search trajectories: token composition',x=.065,ha='left',fontsize=17,weight='bold');fig.text(.065,.90,'500 uniformly sampled trajectories per dataset  |  Qwen3.5-9B tokenizer  |  System messages excluded',fontsize=10,color='#525c69');fig.text(.065,.015,'URL, number and markup categories are rule-based and mutually exclusive; bars show 95% bootstrap intervals.\nTool responses embedded in user messages are separated where explicitly tagged. Tokenized message content only; no chat wrappers.',fontsize=9,color='#525c69')
fig.subplots_adjust(left=.14,right=.97,top=.79,bottom=.23,wspace=.15);fig.savefig(P/'token-composition.png',dpi=180,bbox_inches='tight');fig.savefig(P/'token-composition.pdf',bbox_inches='tight');(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
