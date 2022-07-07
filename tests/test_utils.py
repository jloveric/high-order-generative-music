import pytest
from high_order_generative_music.utils import generate_audio, AudioGenerationSampler
from high_order_generative_music.networks import Net
from omegaconf import DictConfig
import torch
from pytorch_lightning import Trainer


def make_config(features: int):
    cfg = DictConfig(
        {
            "filename": "music/TeaKPea-vpunk.mp3",
            "n": 3,
            "conv": {
                "layer_type": "continuous1d",
                "n": 3,
                "periodicity": 2,
                "kernel_size": [10, 10, 10, 10],
                "stride": [10, 10, 10, 10],
                "channels": [1, 2, 4, 16],
                "segments": 2,
            },
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
                "scheduler": "plateau",
                "patience": 10,
                "factor": 0.1,
            },
            "data": {"window_size": features, "output_window_size": 1},
            "max_epochs": 50,
            "gpus": 1,
            "lr": 0.001,
            "batch_size": 512,
            "train": True,
            "checkpoint": None,
        }
    )
    return cfg


def test_generate_audio():

    features = 1000

    cfg = make_config(features)

    model = Net(cfg)

    samples = 2
    output_size = 100

    result = generate_audio(
        model=model, features=features, samples=samples, output_size=output_size
    )

    assert result.shape == torch.Size([2, 1, 1100])


def test_audio_generation_sampler():

    features = 1000

    cfg = make_config(features)

    model = Net(cfg)
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        gpus=cfg.gpus,
    )

    # Just make sure this runs
    generator = AudioGenerationSampler(
        features=1000, samples=2, output_size=100, sample_rate=2000
    )
    generator.on_train_epoch_end(trainer=trainer, pl_module=model)
