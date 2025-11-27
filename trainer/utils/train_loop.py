import logging
import weakref

class HookBase:
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

class TrainerBase:
    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)

        self._hooks.extend(hooks)

    def register_model(self, model):
        model.trainer = weakref.proxy(self)

    def train(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError
