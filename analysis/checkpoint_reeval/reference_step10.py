"""CPU reference in the SFT environment, with original step-10 tensors."""
import json, os
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from verl.utils.model import get_hf_auto_model_class
from safetensors.torch import load_file

torch.set_num_threads(4)
base = '/root/autodl-tmp/models/Qwen3.5-9B'
root = Path('/root/autodl-tmp/Glasseek')
rawpath = root/'checkpoints/glasseek/sft_128k_length_curriculum_20261104/global_step_10/model_world_size_1_rank_0.pt'
adapter = root/'analysis/checkpoint_reeval/adapters/step_10/adapter_model.safetensors'
print('Loading original checkpoint and checking conversion', flush=True)
os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29791', RANK='0', WORLD_SIZE='1')
torch.distributed.init_process_group('gloo')
raw = torch.load(rawpath, map_location='cpu', weights_only=False)
converted = load_file(str(adapter))
original = {}
for k, v in raw.items():
    if hasattr(v, 'to_local'): v = v.to_local()
    nk = k.replace('.lora_A.default.weight','.lora_A.weight').replace('.lora_B.default.weight','.lora_B.weight')
    if nk not in converted or not torch.equal(v.cpu(), converted[nk]): raise RuntimeError(f'conversion differs: {k}')
    original[k] = v.cpu()
if len(original) != len(converted): raise RuntimeError('adapter key count differs')
del raw, converted
print(f'{len(original)} original tensors match converted weights', flush=True)
config = AutoConfig.from_pretrained(base, trust_remote_code=True)
model = get_hf_auto_model_class(hf_config=config).from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='eager', low_cpu_mem_usage=True)
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules='all-linear', bias='none'))
state = model.state_dict()
if any(k not in state or state[k].shape != v.shape for k,v in original.items()):
    raise RuntimeError('original LoRA tensor names or shapes do not match training architecture')
with torch.no_grad():
    for k,v in original.items(): state[k].copy_(v)
# verl casts trainable LoRA tensors to the bf16 base dtype after wrapping.
model.to(torch.bfloat16)
model.eval()
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
ids = tok('The capital of France is', return_tensors='pt').input_ids
print(f'Computing reference for token ids {ids.tolist()}', flush=True)
with torch.inference_mode():
    with model.disable_adapter(): base_logits = model(input_ids=ids).logits[0,-1].float()
    lora_logits = model(input_ids=ids).logits[0,-1].float()
base_logprobs = base_logits.log_softmax(-1); lora_logprobs = lora_logits.log_softmax(-1)
indices = torch.topk(lora_logprobs, 20).indices.tolist()
result = {'input_ids':ids[0].tolist(), 'top_token_ids':indices,
          'base_logprobs':[base_logprobs[i].item() for i in indices], 'lora_logprobs':[lora_logprobs[i].item() for i in indices],
          'max_abs_logit_delta': (lora_logits-base_logits).abs().max().item(), 'original_tensor_count':len(original)}
if result['max_abs_logit_delta'] < 1e-4: raise RuntimeError('Adapter has no measurable effect')
out = root/'analysis/checkpoint_reeval/step10_reference.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print('Saved CPU reference', out, flush=True)
