import mlflow
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf
from high_order_generative_music.plotting import plot_waveform, plot_specgram
from high_order_generative_music.data import SingleRecordingDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from high_order_generative_music.networks import Net
import logging
import os

logging.basicConfig()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="memorize")
def memorize(cfg: DictConfig):

    base_dir = utils.get_original_cwd()
    mlflow.set_tracking_uri(f"file://{base_dir}/mlruns")

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    if cfg.train is True:
        full_path = f"{root_dir}/{cfg.filename}"
        datamodule = SingleRecordingDataModule(
            filename=full_path,
            batch_size=cfg.batch_size,
            window_size=cfg.data.window_size,
            output_window_size=cfg.data.output_window_size,
        )
        """
        audio_generator = AudioGenerator(
            filename=full_path[0], batch_size=cfg.batch_size
        )
        """
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = Trainer(
            max_epochs=cfg.max_epochs, gpus=cfg.gpus, callbacks=[lr_monitor]
        )
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            callbacks=[lr_monitor],
        )
        model = Net(cfg)
        trainer.fit(model, datamodule=datamodule)
        logger.info("testing")

        trainer.test(model, datamodule=datamodule)
        logger.info("finished testing")
        logger.info("best check_point", trainer.checkpoint_callback.best_model_path)
    else:
        # plot some data
        logger.info("evaluating result")
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{root_dir}/{cfg.checkpoint}"

        logger.info(f"checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)

        model.eval()
        image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[0]}"


if __name__ == "__main__":
    memorize()
