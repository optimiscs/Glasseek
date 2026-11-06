import json,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family']='WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus']=False
base=Path('/root/autodl-tmp/Glasseek/data/glasseek/sft-eval-aligned-128k')
rows=[json.loads(s) for s in (base/'audit.jsonl').open()];kept=[x for x in rows if 'original_system_tokens' in x]
counts=collections.Counter('保留' if not x['reasons'] else '删除：额外工具' if 'out_of_whitelist_tool' in x['reasons'] else '删除：参数不合法' if 'invalid_tool_parameters' in x['reasons'] else '删除：格式异常' for x in rows)
old=np.array([x['original_system_tokens'] for x in kept]);new=np.array([x['aligned_system_tokens'] for x in kept])
fig,(a,b)=plt.subplots(1,2,figsize=(14,5.4),layout='constrained')
names=list(counts);vals=[counts[k] for k in names];bars=a.bar(names,vals,color=['#1c8a77' if n=='保留' else '#cc7257' for n in names]);a.bar_label(bars,fmt='%d',padding=4);a.set_ylabel('轨迹数');a.set_title('完整 5,762 条原始轨迹的筛选结果');a.set_ylim(0,max(vals)*1.2);a.tick_params(axis='x',rotation=10)
b.hist(old,bins=45,alpha=.62,label=f'修改前：均值 {old.mean():,.0f}',color='#5683b8');b.hist(new,bins=45,alpha=.72,label=f'修改后：均值 {new.mean():,.0f}',color='#de854f');b.set_xlabel('System prompt 长度（Qwen token）');b.set_ylabel('轨迹数');b.set_title('保留轨迹：修改前后长度分布');b.legend();fig.suptitle('SFT 轨迹评估对齐：工具筛选与 System 长度',fontsize=15)
fig.savefig(base/'comparison.png',dpi=180);fig.savefig(base/'comparison.pdf');print('wrote comparison.png and comparison.pdf')
