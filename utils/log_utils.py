import os
import sys
import logging
import numpy as np
from datetime import datetime
import torch.distributed as dist

class AverageMeter(object):
    def __init__(self, length=0, fstr=""):
        self.length = length
        self.fstr = fstr

        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        dist.all_reduce(tensor)
        self.update(tensor.item(), num=num)

    def reduce_update_group(self, tensor, num=1, group=None):
        if not group:
            dist.all_reduce(tensor)
        else:
            dist.all_reduce(tensor, group=group)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    def get_val_str(self):
        if len(self.fstr) > 0 and self.fstr.startswith("%"):
            return self.fstr % self.val
        else:
            return str(self.val)

    def get_avg_str(self):
        if len(self.fstr) > 0 and self.fstr.startswith("%"):
            return self.fstr % self.avg
        else:
            return str(self.avg)


class LOGGER(logging.Logger):
    def __init__(self, logger_name):
        super(LOGGER, self).__init__(logger_name)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s")
        console.setFormatter(formatter)
        self.addHandler(console)
        self.rank = 0

    def setup_logging_file(self, log_dir, rank=0):
        self.rank = rank
        if self.rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_name = datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S") + ".log"
            log_fn = os.path.join(log_dir, log_name)
            fh = logging.FileHandler(log_fn)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            fh.setFormatter(formatter)
            self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info("Args:")
        if isinstance(args, (list, tuple)):
            for value in args:
                self.info("--> {}".format(value))
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                self.info("--> {}: {}".format(key, args_dict[key]))
        self.info("")

logger_name = "janus-sft"
logger = LOGGER(logger_name)
