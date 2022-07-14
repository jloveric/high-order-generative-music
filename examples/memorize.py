# import mlflow
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


@hydra.main(config_path="../configs", config_name="memorize")
def memorize(cfg: DictConfig):

    base_dir = utils.get_original_cwd()

    # mlflow_runs = f"file://{base_dir}/mlruns"
    # mlflow.set_tracking_uri(mlflow_runs)

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    full_path = f"{root_dir}/{cfg.filename}"
    datamodule = SingleRecordingDataModule(
        filename=full_path,
        batch_size=cfg.batch_size,
        window_size=cfg.data.window_size,
        output_window_size=cfg.data.output_window_size,
        max_size=cfg.data.max_size,
    )

    audio_generator = AudioGenerationSampler(
        features=cfg.data.window_size,
        samples=2,
        output_size=cfg.sample.audio_size,
        sample_rate=datamodule.sample_rate,
    )
    waveform_generator = WaveformImageSampler(
        features=cfg.data.window_size,
        samples=2,
        output_size=cfg.sample.waveform_size,
        sample_rate=datamodule.sample_rate,
    )

    # Logging for both tensorboard and mlflow
    # mlflow is not doing what I want...
    # multilogger = MultiLogger(mlflow_path=mlflow_runs)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        gpus=cfg.gpus,
        callbacks=[lr_monitor, audio_generator, waveform_generator],
    )

    model = Net(cfg)
    trainer.fit(model, datamodule=datamodule)
    logger.info("testing")

    trainer.test(model, datamodule=datamodule)
    logger.info("finished testing")
    logger.info(f"best check_point {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    memorize()
