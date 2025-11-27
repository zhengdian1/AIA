from trainer.utils import *
from .trainer_t2i import TextToImageTrainer
from .trainer_editing import ImageEditingTrainer

def setup_task(cfg, is_root):
    if cfg.common.task == 't2i':
        Trainer = TextToImageTrainer(cfg)
        hook_list = [
            LRScheduler(
                Trainer.optimizer,
                cfg.optimize.lr,
                cfg.optimize.warmup_epochs * cfg.dataloader_len,
                cfg.optimize.lr_scheduler,
            ),
            UniversalMeterLogger(cfg),
            SaveModel(cfg, is_root=is_root),
        ]
    elif cfg.common.task == 'editing':
        Trainer = ImageEditingTrainer(cfg)
        hook_list = [
            LRScheduler(
                Trainer.optimizer,
                cfg.optimize.lr,
                cfg.optimize.warmup_epochs * cfg.dataloader_len,
                cfg.optimize.lr_scheduler,
            ),
            UniversalMeterLogger(cfg),
            SaveModel(cfg, is_root=is_root),
        ]
    else:
        raise NotImplementedError

    return Trainer, hook_list
