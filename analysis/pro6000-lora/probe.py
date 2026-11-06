"""Single-GPU LoRA probe; no FSDP, one real trajectory per rank, one update."""
import json, os, time, sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, TaskType, get_peft_model
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils import hf_processor, hf_tokenizer
from omegaconf import OmegaConf
from glasseek_training.sft_dataset import GlasseekSFTDataset
import pyarrow as pa, pyarrow.parquet as pq

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl')
source_line=int(sys.argv[1]); root=Path('/root/autodl-tmp/Glasseek')
model_path='/root/autodl-tmp/models/Qwen3.5-9B'
probe=root/f'analysis/ddp-lora-probe/line-{source_line}.parquet'
tok=hf_tokenizer(model_path);processor=hf_processor(model_path)
if processor is not None and not processor.chat_template: processor.chat_template=tok.chat_template
cfg=OmegaConf.create({'pad_mode':'no_padding','max_length':327680,'truncation':'error','messages_key':'messages','ignore_input_ids_mismatch':True})
data=GlasseekSFTDataset(parquet_files=str(probe),tokenizer=tok,processor=processor,config=cfg)[0]
ids=data['input_ids'].to(rank).unsqueeze(0); mask=data['loss_mask'].to(rank).roll(-1).unsqueeze(0)
pos=data['position_ids'].to(rank)
pos=pos.unsqueeze(1) if pos.ndim==2 else pos.unsqueeze(0)
print(f'DDP_DATA rank={rank} length={ids.shape[-1]} supervised={int(mask.sum())}',flush=True)
t0=time.monotonic()
try:
 model=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16,
                   attn_implementation='flash_attention_2',local_files_only=True)
 model.config.use_cache=False
 apply_monkey_patch(model,ulysses_sp_size=1,use_remove_padding=True,use_fused_kernels=True,fused_kernels_backend='torch')
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
                                      target_modules='all-linear',bias='none'))
 for param in model.parameters():
  if param.requires_grad and param.dtype != torch.bfloat16:param.data=param.data.to(torch.bfloat16)
 model=model.to(rank)
 model=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-5,weight_decay=.01)
 model.train()
 torch.cuda.synchronize()
 print(f'DDP_MODEL rank={rank} allocated_gib={torch.cuda.memory_allocated()/2**30:.3f}',flush=True)
 result=model(input_ids=ids,position_ids=pos,attention_mask=None,use_cache=False)
 loss=-(result.log_probs*mask).sum()/mask.sum()
 loss.backward()
 grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
 opt.step();torch.cuda.synchronize()
 print('DDP_RESULT',json.dumps({'rank':rank,'tokens':ids.shape[-1],'loss':float(loss.detach()),
       'grad_norm':float(grad_norm),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
       'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'elapsed_seconds':time.monotonic()-t0}),flush=True)
except Exception as e:
 print('DDP_FAILURE',json.dumps({'rank':rank,'tokens':ids.shape[-1],'error_type':type(e).__name__,
       'error':str(e)[:800],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
       'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}),flush=True)
 raise
finally:
 dist.destroy_process_group()
