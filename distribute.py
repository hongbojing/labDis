import os
import torch.distributed as dist


def init_processes(worker, n_workers, fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=worker, world_size=n_workers)
    fn(worker, n_workers)