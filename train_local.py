import os
import time
import torch
import datetime
import numpy as np
import torch.distributed as dist
from utils import logger
from trainer import setup_task
from trainer.utils import set_seed, parse_args, parse_args_from_yaml

os.environ["COMBINED_ENABLE"] = "1"
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    args.DEVICE_TYPE = DEVICE_TYPE
    cfg_yml = parse_args_from_yaml(args.yml_path)
    cfg = args
    cfg.update(cfg_yml)
    set_seed(cfg.common.random_seed)
    is_root = (cfg.train_data_index in list(range(cfg.common.machines))) and (args.rank % 8 == 0)

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(hours=2.0),
    )

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(args.rank % num_gpus)
    now_time = torch.from_numpy(np.array(int(time.time()))).float().cuda()
    dist.all_reduce(now_time)
    
    cfg.common.output_path = os.path.join(cfg.common.output_path, cfg.yml_path, str(now_time.cpu().numpy()))
    cfg.common.log_path = os.path.join(cfg.common.log_path, cfg.yml_path, str(now_time.cpu().numpy()))

    logger.setup_logging_file(cfg.common.log_path, cfg.rank)
    Trainer, hook_list = setup_task(cfg=cfg, is_root=is_root)
    Trainer.register_hooks(hook_list)
    Trainer.register_model(Trainer.model.module if hasattr(Trainer.model, "module") else Trainer.model)
    max_iter = int(cfg.optimize.max_epochs * cfg.dataloader_len)
    Trainer.train(0, max_iter)