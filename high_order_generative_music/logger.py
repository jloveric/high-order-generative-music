from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger


class MultiLogger(LightningLoggerBase):
    def __init__(self, mlflow_path: str):
        save_dir = "lightning_logs"
        self.tf_logger = TensorBoardLogger(save_dir=save_dir)
        self.mlflow_logger = MLFlowLogger(experiment_name=mlflow_path)

    @property
    def name(self):
        return "multilogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.tf_logger.log_hyperparams(params)
        self.mlflow_logger.log_hyperparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.tf_logger.log_metrics(metrics, step)
        self.mlflow_logger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        self.tf_logger.save()
        self.mlflow_logger.save()

    @rank_zero_only
    def finalize(self, status):
        self.tf_logger.finalize(status)
        self.mlflow_logger.finalize(status)

    @property
    @rank_zero_experiment
    def experiment(self):
        # Experiment is only defined for tensorflow for now
        return self.tf_logger.experiment
