import mlflow
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf
from high_order_generative_music.plotting import plot_waveform, plot_specgram
from high_order_generative_music.data import SingleRecordingDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from high_order_generative_music.networks import Net
from high_order_generative_music.utils import (
    AudioGenerationSampler,
    WaveformImageSampler,
)
from high_order_generative_music.logger import MultiLogger
from high_order_generative_music.utils import extend_audio

import logging
import os

logging.basicConfig()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="generate")
def memorize(cfg: DictConfig):

    base_dir = utils.get_original_cwd()

    mlflow_runs = f"file://{base_dir}/mlruns"
    mlflow.set_tracking_uri(mlflow_runs)

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    logger.info("evaluating result")
    logger.info(f"cfg.checkpoint {cfg.checkpoint}")
    checkpoint_path = f"{root_dir}/{cfg.checkpoint}"

    logger.info(f"checkpoint_path {checkpoint_path}")
    model = Net.load_from_checkpoint(checkpoint_path)
    print("model.features", model._features)

    model.eval()
    new_audio = extend_audio(model=model, features=model._features, sample=sample)

    image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[0]}"


if __name__ == "__main__":
    generate()
