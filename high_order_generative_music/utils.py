from high_order_layers_torch.layers import *
from pytorch_lightning import Callback
from high_order_layers_torch.networks import *
from torch import nn
import logging

logger = logging.getLogger(__name__)


def generate_audio(model: nn.Module, features: int, samples: int, output_size: int):

    model.eval()
    features = features
    values = torch.rand(samples, 1, features, device=model.device) * 2 - 1

    for i in range(output_size):
        model.eval()
        output = model(values[:, :, -features:])
        output = output.unsqueeze(1)
        values = torch.cat([values, output], dim=2)

    return values


class AudioGenerationSampler(Callback):
    def __init__(self, features: int, samples: int, output_size: int, sample_rate: int):
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
