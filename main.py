from torch.multiprocessing import Process

from distribute import init_processes
from train import train
import hps


n_workers = hps.n_workers
processes = []
for worker in range(n_workers):
    p = Process(target=init_processes, args=(worker, n_workers, train))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
