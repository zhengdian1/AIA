import os
import torch
import argparse
import multiprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:51683")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--args_yml_fn", type=str, default="configs/t2i_generation.yml")
    args, _ = parser.parse_known_args()
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node
    core_per_proc = int(multiprocessing.cpu_count() / ngpus_per_node)
    for rank in range(ngpus_per_node):
        taskset = "taskset -c {}-{}".format(core_per_proc * rank, core_per_proc * (rank + 1) - 1)
        cmd = "export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1; "
        cmd += "OMP_NUM_THREADS=4 {} python train_local.py".format(taskset)
        cmd += " --backend={} --init_method={}".format(args.backend, args.init_method)
        cmd += " --rank={} --world_size={}".format(rank, world_size)
        cmd += " --yml_path={}".format(args.args_yml_fn)
        if rank < ngpus_per_node - 1:
            cmd += " &"
        os.system(cmd)

