"""Compare checkpoint shards with exported safetensors, exactly and on CPU."""
import argparse
import json
from pathlib import Path


def compare_tensor(merged, tensor, rank, world_size):
    import torch
    from torch.distributed.tensor import Shard, Replicate
    local = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
    placements = getattr(tensor, 'placements', ())
    if len(placements) > 1:
        raise ValueError('Unexpected multidimensional mesh')
    if placements and isinstance(placements[0], Shard):
        dim = placements[0].dim
        chunk = (merged.shape[dim] + world_size - 1) // world_size
        start = min(rank * chunk, merged.shape[dim])
        view = merged.narrow(dim, start, local.shape[dim])
    elif not placements or isinstance(placements[0], Replicate):
        view = merged
    else:
        raise ValueError(f'Unsupported placement: {placements}')
    return view.shape == local.shape and view.dtype == local.dtype and torch.equal(view, local)


def verify(source, target, world_size=2):
    import torch
    from safetensors import safe_open
    if world_size < 1:
        raise ValueError('world_size must be positive')
    checked = 0
    with safe_open(target, framework='pt', device='cpu') as exported:
        keys = set(exported.keys())
        for rank in range(world_size):
            state = torch.load(Path(source) / f'model_world_size_{world_size}_rank_{rank}.pt', map_location='cpu', weights_only=False)
            if set(state) != keys:
                raise ValueError('State key mismatch')
            for name, tensor in state.items():
                if not compare_tensor(exported.get_tensor(name), tensor, rank, world_size):
                    raise ValueError(f'Weight mismatch: rank={rank}, key={name}')
                checked += 1
            del state
    return {'status': 'passed', 'source_shards': world_size, 'keys': len(keys),
            'tensor_shards_checked': checked, 'comparison': 'exact dtype and values'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    print(json.dumps(verify(args.source, args.target, args.world_size)))


if __name__ == '__main__':
    main()
