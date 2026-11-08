"""One DDP LoRA probe with explicit context-length and activation-offload profiles."""
import argparse
import json
from pathlib import Path
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]


def configuration(profile, overrides=None):
    document = json.loads((ROOT / 'RL_training/configs/lora_probe_profiles.json').read_text())
    result = dict(document['defaults'], **document['profiles'][profile])
    result.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if result['max_length'] < 1 or result['lora_rank'] < 1:
        raise ValueError('Lengths and LoRA rank must be positive')
    return result


def run(args, config):
    import json, os, time
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

    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    source_line=args.source_line; root=Path(__file__).resolve().parents[2]
    model_path=config['model_path']
    probe=args.data_path or root/f'experiments/ddp-lora-probe/line-{source_line}.parquet'
    tok=hf_tokenizer(model_path);processor=hf_processor(model_path)
    if processor is not None and not processor.chat_template: processor.chat_template=tok.chat_template
    cfg=OmegaConf.create({'pad_mode':'no_padding','max_length':config['max_length'],'truncation':config['truncation'],'messages_key':'messages','ignore_input_ids_mismatch':True})
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
     model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=config['lora_rank'],lora_alpha=config['lora_alpha'],
                                          target_modules='all-linear',bias='none'))
     for param in model.parameters():
      if param.requires_grad and param.dtype != torch.bfloat16:param.data=param.data.to(torch.bfloat16)
     model=model.to(rank)
     model=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
     opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['lr'],weight_decay=config['weight_decay'])
     model.train()
     torch.cuda.synchronize()
     print(f'DDP_MODEL rank={rank} allocated_gib={torch.cuda.memory_allocated()/2**30:.3f}',flush=True)
     with torch.autograd.graph.save_on_cpu(pin_memory=True) if config['activation_offload'] else nullcontext():
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_line', type=int)
    parser.add_argument('--profile', default='pro6000', choices=('pro6000', '128k', 'offload', 'ddp'))
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--model-path')
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--truncation', choices=('error', 'right'))
    parser.add_argument('--activation-offload', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--lora-rank', type=int)
    parser.add_argument('--lora-alpha', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args()
    keys = ('model_path', 'max_length', 'truncation', 'activation_offload', 'lora_rank', 'lora_alpha', 'lr', 'weight_decay')
    config = configuration(args.profile, {key: getattr(args, key) for key in keys})
    if args.print_config:
        print(json.dumps(config))
        return
    run(args, config)


if __name__ == '__main__':
    main()
