import mlflow
import hydra
from hydra import utils

mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')