from typing import List

from high_order_layers_torch.layers import *
from pytorch_lightning import Callback
from high_order_layers_torch.networks import *
from language_interpolation.single_text_dataset import (
    encode_input_from_text,
    decode_output_to_text,
    ascii_to_float,
)
from torch import nn
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_audio(model: nn.Module, features: int, samples: int, output_size: int):

    model.eval()
    features = features
    sample_start = torch.rand(samples, 1, features, device=model.device) * 2 - 1

    values = sample_start
    for i in range(output_size):
        model.eval()
        output = model(values[:, :, -features:])
        values = torch.stack([values, output])

    return values


class AudioGenerationSampler(Callback):
    def __init__(self, features, samples, output_size, sample_rate: int):
        super().__init__()
        self._features = features
        self._samples = samples
        self._output_size = output_size
        self._sample_rate = sample_rate

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):

        audio = generate_audio(
            model=pl_module,
            features=self._features,
            samples=self._samples,
            output_size=self._output_size,
        )

        trainer.logger.experiment.add_audio(
            f"audio",
            audio,
            sample_rate=self._sample_rate,
            global_step=trainer.global_step,
        )
