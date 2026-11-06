import torch,json,sys
from safetensors import safe_open
from torch.distributed.tensor import Shard,Replicate
from pathlib import Path
root=Path('/root/autodl-tmp/Glasseek');src=root/'checkpoints/glasseek/sft_64k_full_2gpu_turns/global_step_76';target=root/'analysis/full64k-reeval/step76-hf/model.safetensors'
torch.set_num_threads(4); checked=0
with safe_open(target,framework='pt',device='cpu') as f:
 allkeys=set(f.keys())
 for rank in range(2):
  state=torch.load(src/f'model_world_size_2_rank_{rank}.pt',map_location='cpu',weights_only=False)
  if set(state)!=allkeys:raise RuntimeError('state key mismatch')
  for i,(name,tensor) in enumerate(state.items()):
   local=tensor.to_local() if hasattr(tensor,'to_local') else tensor
   merged=f.get_tensor(name); placements=getattr(tensor,'placements',())
   if len(placements)>1:raise RuntimeError('unexpected mesh')
   if placements and isinstance(placements[0],Shard):
    dim=placements[0].dim; size=merged.shape[dim]; chunk=(size+1)//2; start=rank*chunk; view=merged.narrow(dim,start,local.shape[dim])
   elif not placements or isinstance(placements[0],Replicate):view=merged
   else:raise RuntimeError(f'unsupported placement {name} {placements}')
   if view.shape!=local.shape or view.dtype!=local.dtype or not torch.equal(view,local):raise RuntimeError(f'weight mismatch {rank} {name}')
   checked+=1
   if (i+1)%100==0:print('rank',rank,'checked',i+1,flush=True)
  del state
print(json.dumps({'status':'passed','source_shards':2,'keys':len(allkeys),'tensor_shards_checked':checked,'comparison':'exact dtype and values'}),flush=True)
