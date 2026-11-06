from pathlib import Path
import json,collections,re
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
plt.rcParams.update({'font.family':'WenQuanYi Zen Hei','axes.unicode_minus':False,'font.size':11,'savefig.dpi':170})
root=Path('/root/autodl-tmp/Glasseek');out=Path(__file__).parent; tb=root/'analysis/sft128k/tensorboard_comparison'
def read(run,tag):
 e=EventAccumulator(str(tb/run),size_guidance={'scalars':0},purge_orphaned_data=False);e.Reload();vals={x.step:x.value for x in e.Scalars(tag)};return sorted(vals.items())
fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
for run,label,col in [('full_64k_2gpu','双 PRO6000 全参 64K','#22577a'),('lora_128k_1gpu','单 PRO6000 LoRA 128K','#e07a5f')]:
 for ax,tag in zip(axs,['train/loss','perf/max_memory_allocated_gb']):
  v=read(run,tag);ax.plot([x for x,y in v],[y for x,y in v],label=label,color=col,linewidth=1.8)
axs[0].set(title='训练 loss（不同样本，仅看趋势）',xlabel='优化器 step',ylabel='训练 loss');axs[1].set(title='训练过程 GPU 已分配显存峰值',xlabel='优化器 step',ylabel='GiB / GPU');
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.savefig(out/'training_curves.png');plt.close(fig)
v=read('lora_128k_1gpu','val/loss');fig,ax=plt.subplots(figsize=(7,4.1),layout='constrained');ax.plot([x for x,y in v],[y for x,y in v],marker='o',color='#e07a5f');
for x,y in v:ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set(title='LoRA 128K：历史检查点验证 loss',xlabel='检查点 step',ylabel='验证 loss');ax.grid(alpha=.2);fig.savefig(out/'validation.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained');names=['原始 SFT','工具对齐后','其中可用于\n64K 训练/验证'];values=[5762,3666,3265];bars=axs[0].bar(names,values,color=['#9db4c0','#48a999','#22577a']);axs[0].bar_label(bars,padding=3);axs[0].set_ylim(0,6500);axs[0].set(title='数据过滤与规模',ylabel='轨迹数')
a=1310.57;b=1230.84;bars=axs[1].bar(['切换前首步','快速模式首步'],[a/60,b/60],color=['#9db4c0','#48a999']);axs[1].bar_label(bars,labels=['21分50秒','20分31秒'],padding=3);axs[1].set(title='A100：同首批次初步对照',ylabel='分钟 / step',ylim=(0,26));fig.savefig(out/'data_and_speed.png');plt.close(fig)
p=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets');names=[];yes=[];no=[];remaining=[]
for name,folder in [('基座','glasseek-base9b-rl9b-34')]+[(f'step{i}',f'glasseek-step{i}-rl9b-34') for i in range(10,81,10)]:
 f=p/folder/'traces.json'
 if not f.exists():continue
 cases=json.loads(f.read_text())['cases'];n=sum(z.get('evaluationStatus')=='correct' for z in cases);m=sum(z.get('evaluationStatus')=='incorrect' for z in cases);names.append(name);yes.append(n);no.append(m);remaining.append(34-n-m)
fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained');ax.bar(names,yes,label='已判正确',color='#48a999');ax.bar(names,no,bottom=yes,label='已判错误',color='#e07a5f');ax.bar(names,remaining,bottom=[a+b for a,b in zip(yes,no)],label='未完成 / 缺失 / 异常',color='#d2d9df');ax.set(title='BrowseComp 34 题：历史快照，任务未全部完成',ylabel='题数',ylim=(0,37));ax.legend(ncol=3,fontsize=9);fig.savefig(out/'eval_incomplete.png');plt.close(fig)
print('charts created:',*[x.name for x in out.glob('*.png')])
