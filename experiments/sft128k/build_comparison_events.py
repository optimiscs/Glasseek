from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
base=Path('/root/autodl-tmp/Glasseek/experiments/sft128k');target=base/'tensorboard_comparison/lora_128k_1gpu';target.mkdir(parents=True,exist_ok=True)
writer=SummaryWriter(str(target))
for folder,choose in [('length_curriculum',lambda k:k!='train/total_tokens(B)'),('length_curriculum_total_tokens_corrected',lambda k:k=='train/total_tokens(B)'),('length_curriculum_validation_history',lambda k:k=='val/loss')]:
 e=EventAccumulator(str(base/'tensorboard'/folder),size_guidance={'scalars':0},purge_orphaned_data=False);e.Reload()
 for tag in e.Tags()['scalars']:
  if not choose(tag):continue
  values={}
  for x in e.Scalars(tag):
   if 1<=x.step<=80 and (x.step not in values or x.wall_time>values[x.step].wall_time):values[x.step]=x
  for step,x in sorted(values.items()):writer.add_scalar(tag,x.value,step,walltime=x.wall_time)
  print(folder,tag,len(values),flush=True)
writer.flush();writer.close()
