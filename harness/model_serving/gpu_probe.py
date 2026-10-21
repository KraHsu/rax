"""Run under torchrun inside the serving container, using its existing allocation."""
import json
import os
from datetime import timedelta


def main():
    import torch
    import torch.distributed as dist
    import vllm

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    try:
        count = dist.get_world_size()
        value = torch.tensor([rank + 1.0], device='cuda')
        dist.all_reduce(value)
        if value.item() != count * (count + 1) / 2:
            raise RuntimeError('GPU collective returned an incorrect result')
        host = torch.ones(1024, pin_memory=True)
        copied = host.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        if copied.sum().item() != 1024:
            raise RuntimeError('Pinned host-memory transfer returned an incorrect result')
        if rank == 0:
            print(json.dumps({'gpu_preflight': 'passed', 'gpus': count,
                              'device': torch.cuda.get_device_name(rank), 'vllm': vllm.__version__,
                              'nccl_all_reduce': True, 'pinned_host_transfer': True}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
