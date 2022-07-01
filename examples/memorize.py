import mlflow
import hydra
from hydra import utils
from omegaconf import DictConfig

mlflow.set_tracking_uri(f"file://{utils.get_original_cwd()}/mlruns")


@hydra.main(config_path="../configs", config_name="memorize")
def memorize(cfg: DictConfig):
    pass


if __name__ == "__main__":
    memorize()
