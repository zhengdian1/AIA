import os
import yaml
import torch
import random
import numpy as np

from .train_loop import HookBase
from utils import logger

def set_seed(seed, mode=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = False
    torch.backends.cudnn.benchmark = False

def print_model_param_num(model_info, model):
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(
                "\nmodel_info:\n{}\ntotal_params: {}\ntrainable_params: {}\n".format(
                    model_info, params_total / 1024 / 1024, params_trainable / 1024 / 1024
                )
            )
    else:
        print(
            "\nmodel_info:\n{}\ntotal_params: {}\ntrainable_params: {}\n".format(
                model_info, params_total, params_trainable
            )
        )

class SaveModel(HookBase):
    def __init__(self, cfg, is_root):
        self.output_path = cfg.common.output_path
        self.is_root = is_root
        if cfg.common.get("save_per_iters", 0)>0:
            self.save_interval = cfg.common.get("save_per_iters", 0)
        else:
            self.save_interval = int(cfg.common.save_per_epochs * cfg.dataloader_len)

        self._cfg = cfg
        self.delete_after_upload = cfg.common.get("delete_after_upload", False)
        self.only_save_lora = cfg.common.get("only_save_lora", False)

    @property
    def save_path(self):
        if self.__dict__.get("_output_path", None) is None:
            output_path = os.path.join(self.output_path, "ckpt")
            os.makedirs(output_path, exist_ok=True)
            self.__dict__["_output_path"] = output_path
        return self.__dict__["_output_path"]

    def before_train(self):
        self.save_config()

    def after_step(self):
        try:
            if self.trainer.iter % self.save_interval == (self.save_interval - 1):
                save_name = "iter_%d.pth" % (self.trainer.iter)
                if self.is_root:
                    print('save_model to:', os.path.join(self.save_path, save_name))
                if self._cfg.common.use_fsdp:
                    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(self.trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                        cpu_state = self.trainer.model.state_dict()
                        self.save_model(cpu_state, save_name)
                torch.cuda.empty_cache()

        except Exception as e:
            print(e)

    def save_config(self):
        if self.is_root:
            logger.save_args(self._cfg)
            run_save_path = os.path.join(self.output_path, "run.yml")
            if not os.path.isfile(run_save_path):
                try:
                    dir_path = os.path.dirname(run_save_path)
                    os.makedirs(dir_path, exist_ok=True)
                    with open(run_save_path, "w") as args_fh:
                        yaml.dump(self._cfg.__dict__, args_fh, sort_keys=False)
                    logger.info("Run configs dump to %s" % run_save_path)
                except Exception as e:
                    print(e)
                    logger.info("fail to dump run config!!")
    
    def save_model(self, checkpoint, save_name):
        local_weights = os.path.join(self.save_path, save_name)
        import torch.distributed as dist
        dist.barrier()
        if dist.get_rank() == 0:
            cpu_ckpt = {k: v.cpu() for k, v in checkpoint.items()}
            torch.save(cpu_ckpt, local_weights)
        dist.barrier()
