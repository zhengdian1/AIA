import time
import datetime
from .train_loop import HookBase
from utils import logger

class UniversalMeterLogger(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg

    def before_train(self):
        self._train_time = 0

    def after_train(self):
        logger.info("Finish training, total time: {}".format(str(datetime.timedelta(seconds=int(self._train_time)))))

    def before_step(self):
        self._start_time = time.perf_counter()

    def after_step(self):
        step_time = time.perf_counter() - self._start_time
        self._train_time += step_time
        self.trainer.meters.batch_time.update(step_time)
        current_iter = self.trainer.iter % self.cfg.dataloader_len
        current_epoch = self.trainer.iter // self.cfg.dataloader_len
        if (current_iter + 1) % self.cfg.common.log_interval == 0:
            data_cnt = (
                self.cfg.dataloader.train.batch_size if self.cfg.common.task == "editing" else self.cfg.dataloader.train.task1.batch_size
                * self.trainer.dist.world_size
                * self.trainer.gradient_accumulation_steps
            )
            fps = data_cnt / step_time
            meters = self.trainer.meters
            totals = self.trainer.totals
            total_iters = totals.total_iters
            remain_secs = (total_iters - self.trainer.iter) * meters.batch_time.avg
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
            time_str = f"\tRemainingTime {remain_time} ({finish_time})"
            msg = f"Iter: [{current_iter+1}/{totals.iter_per_epoch}]"
            msg += f"\tEpoch: [{current_epoch}/{totals.epochs} ({totals.total_iters})]"
            for k in meters.keys():
                msg += f"\t{k} {meters[k].get_val_str()} ({meters[k].get_avg_str()})"
            msg += time_str
            msg += f"\tFPS {fps: .3f}"
            logger.info(msg)
