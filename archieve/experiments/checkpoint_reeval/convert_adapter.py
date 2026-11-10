"""Convert a world-size-one verl LoRA shard to PEFT safetensors format.

The source is read-only. Output still needs a model-level equivalence check.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    args = parser.parse_args()
    meta = json.loads((args.checkpoint / 'lora_train_meta.json').read_text())
    raw = torch.load(args.checkpoint / 'model_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
    converted = {}
    for name, tensor in raw.items():
        if '.lora_A.default.weight' not in name and '.lora_B.default.weight' not in name:
            raise ValueError(f'Unsupported adapter tensor: {name}')
        if hasattr(tensor, 'to_local'):
            if tensor.device_mesh.mesh.numel() != 1:
                raise ValueError('Expected world-size-one tensor')
            value = tensor.to_local()
        else:
            value = tensor
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f'Incomplete tensor {name}')
        key = name.replace('.lora_A.default.weight', '.lora_A.weight').replace('.lora_B.default.weight', '.lora_B.weight')
        if key in converted:
            raise ValueError(f'Duplicate key: {key}')
        converted[key] = value.detach().cpu().contiguous()
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(converted, str(args.output / 'adapter_model.safetensors'))
    config = {'peft_type': 'LORA', 'task_type': meta['task_type'], 'r': meta['r'],
              'lora_alpha': meta['lora_alpha'], 'target_modules': 'all-linear',
              'lora_dropout': 0.0, 'bias': 'none', 'inference_mode': True,
              'base_model_name_or_path': args.base_model}
    (args.output / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Converted {len(converted)} tensors to {args.output}; verify loaded outputs before use')


if __name__ == '__main__':
    main()
