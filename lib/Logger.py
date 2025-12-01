from configs.config import LOG
import wandb


class NotLogger:
    def log(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LoggerWrapper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.logger = None

    def __enter__(self):
        if LOG:
            self.logger = wandb.init(*self.args, **self.kwargs)
        else:
            self.logger = NotLogger()

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.__exit__(exc_type, exc_val, exc_tb)
