import mlflow
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf
from high_order_generative_music.plotting import plot_waveform, plot_specgram
from high_order_generative_music.networks import Net
from high_order_generative_music.utils import extend_audio
import torchaudio
import logging
import os

logging.basicConfig()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="generate")
def generate(cfg: DictConfig):

    if cfg.checkpoint is None:
        raise ValueError(f"Must define cfg.checkpoint, got {cfg.checkpoint}")

    base_dir = utils.get_original_cwd()
    filepath = f"{base_dir}/{cfg.filepath}"

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    logger.info("evaluating result")
    logger.info(f"cfg.checkpoint {cfg.checkpoint}")
    checkpoint_path = f"{root_dir}/{cfg.checkpoint}"

    logger.info(f"checkpoint_path {checkpoint_path}")
    model = Net.load_from_checkpoint(checkpoint_path)
    logger.info(OmegaConf.to_yaml(model.cfg))
    print("model.features", model.cfg.net.features)

    model.eval()
    new_audio = extend_audio(model=model, features=model.cfg.net.features, sample=None)

    torchaudio.save(
        filepath=filepath, src=new_audio.unsqueeze(0), sample_rate=cfg.frequency
    )


if __name__ == "__main__":
    generate()
