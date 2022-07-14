from high_order_layers_torch.layers import *
from pytorch_lightning import Callback
from high_order_layers_torch.networks import *
from high_order_generative_music.plotting import plot_waveform
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pytorch_lightning as pl
from torch import nn
import matplotlib.pyplot as plt
import logging
import io
import PIL
from torchvision import transforms

logger = logging.getLogger(__name__)


def extend_audio(
    model: nn.Module,
    features: int,
    sample: torch.Tensor,
    output_size: int,
    noiseless: bool = True,
):
    """
    Generate audio from a given sample
    Args :
        features : Number of features used by the network
        sample : Tensor of size features
        output_size :  additional length to generate
    Returns :
        a flat tensor representing the audio signal
    """

    model.eval()
    with torch.no_grad():
        features = features
        values = sample.unsqueeze(0).unsqueeze(0)

        for i in range(output_size):
            output = model(values[:, :, -features:])
            output = output.unsqueeze(1)
            values = torch.cat([values, output], dim=2)

        return values.flatten()


def generate_audio(
    model: nn.Module,
    features: int,
    samples: int,
    output_size: int,
    noiseless: bool = True,
):

    model.eval()
    with torch.no_grad():
        features = features
        values = torch.rand(samples, 1, features, device=model.device) * 2 - 1

        for i in range(output_size):
            output = model(values[:, :, -features:])
            output = output.unsqueeze(1)
            values = torch.cat([values, output], dim=2)

        return values[:, :, features:]


class AudioGenerationSampler(Callback):
    def __init__(
        self, features: int, samples: int, output_size: int, sample_rate: int = 0
    ):
        super().__init__()
        self._features = features
        self._samples = samples
        self._output_size = output_size

        if sample_rate is None or sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self._sample_rate = sample_rate

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):

        audio = generate_audio(
            model=pl_module,
            features=self._features,
            samples=self._samples,
            output_size=self._output_size,
        )

        for i in range(self._samples):
            trainer.logger.experiment.add_audio(
                f"audio_{i}",
                audio[i].flatten(),
                sample_rate=self._sample_rate,
                global_step=trainer.global_step,
            )


class WaveformImageSampler(Callback):
    def __init__(
        self, features: int, samples: int, output_size: int, sample_rate: int = 0
    ):
        super().__init__()
        self._features = features
        self._samples = samples
        self._output_size = output_size

        if sample_rate is None or sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self._sample_rate = sample_rate

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        with torch.no_grad():

            audio = generate_audio(
                model=pl_module,
                features=self._features,
                samples=self._samples,
                output_size=self._output_size,
            )

            # audio has dimension [batch, 1, wave_length]
            plt.clf()
            for e in range(self._samples):
                plt.plot(audio[e, 0, :].cpu())

            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
                transparent=False,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)

            trainer.logger.experiment.add_image(
                f"waveform", image, global_step=trainer.global_step
            )
