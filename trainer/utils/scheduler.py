import math
from .train_loop import HookBase

class LRScheduler(HookBase):
    def __init__(self, optimizer, init_lr, warmup_iter=0, scheduler=None):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.scheduler = scheduler
        self.warmup_iter = warmup_iter

    def before_step(self):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self.calc_learning_rate(self.trainer.iter, param_group["init_lr"])

    def calc_learning_rate(self, iter, init_lr, minimum_lr=1.0e-6):
        if iter < self.warmup_iter:
            new_lr = (iter + 1) / self.warmup_iter * (init_lr - minimum_lr) + minimum_lr
        else:
            if self.scheduler == "linear":
                new_lr = minimum_lr + (init_lr - minimum_lr) * (
                    (self.trainer.max_iter - iter) / (self.trainer.max_iter - self.warmup_iter)
                )
            elif self.scheduler == "cosine":
                new_lr = minimum_lr + 0.5 * (init_lr - minimum_lr) * (
                    1 + math.cos(math.pi * (iter - self.warmup_iter) / (self.trainer.max_iter - self.warmup_iter))
                )
            else:
                new_lr = init_lr
        return new_lr
