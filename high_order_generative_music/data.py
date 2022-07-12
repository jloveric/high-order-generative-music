import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import pytorch_lightning as pl
from typing import Optional
import torchaudio.functional as F
import torchaudio.transforms as T
import logging

logger = logging.getLogger(__name__)


def single_recording_dataset(
    filename: str, resample_rate: int = 11025, low_pass_filter_width: int = 6
):
    with open(filename, "rb") as file:
        waveform, sample_rate = torchaudio.load(file)
        logger.info(f"sample rate {sample_rate}")
        if resample_rate is not None:
            waveform = F.resample(
                waveform=waveform,
                orig_freq=sample_rate,
                new_freq=resample_rate,
                lowpass_filter_width=low_pass_filter_width,
            )
            return waveform, resample_rate

        return waveform, sample_rate


class SingleRecordingDataset(Dataset):
    def __init__(
        self,
        waveform: Tensor,
        sample_rate: float,
        window_size: int = 1000,
        channel: int = 0,
        output_window_size: int = 1,
    ):
        self._waveform = waveform[channel]
        self._sample_rate = sample_rate
        self._window_size = window_size
        self._output_window_size = output_window_size
        self._size = self._waveform.shape[0] - (window_size + 1) + (output_window_size)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):

        offset = idx + self._window_size
        features = self._waveform[idx:offset].unsqueeze(0)
        targets = self._waveform[offset : (offset + self._output_window_size)]
        return features, targets


class SingleRecordingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        filename: str,
        window_size: int,
        output_window_size: int,
        batch_size: int = 32,
        num_workers: int = 10,
        dataset: Dataset = SingleRecordingDataset,
        channel: int = 0,
        max_size: int = None,
    ):
        super().__init__()
        self._filename = filename
        self._window_size = window_size
        self._output_window_size = output_window_size
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataset = dataset
        self._channel = channel
        self._sample_rate = None
        self._max_size = max_size
        self._initialized = False

        # Currently doing this so sample_rate is computed immediately
        self.setup()

    def setup(self, stage: Optional[str] = None):

        if self._initialized is False:
            # TODO: the sample rate should be set as input and then data should be
            # subsampled.
            waveform, self._sample_rate = single_recording_dataset(
                filename=self._filename
            )
            if self._max_size is not None:
                waveform = waveform[:, : self._max_size]

            self._train_dataset = self._dataset(
                waveform=waveform,
                sample_rate=self._sample_rate,
                window_size=self._window_size,
                output_window_size=self._output_window_size,
                channel=self._channel,
            )
            self._val_dataset = self._dataset(
                waveform=waveform,
                sample_rate=self._sample_rate,
                window_size=self._window_size,
                output_window_size=self._output_window_size,
                channel=self._channel,
            )
            self._test_dataset = self._dataset(
                waveform=waveform,
                sample_rate=self._sample_rate,
                window_size=self._window_size,
                output_window_size=self._output_window_size,
                channel=self._channel,
            )

            logger.info(f"Train dataset size is {len(self._train_dataset)}")
            logger.info(f"Validation dataset size is {len(self._val_dataset)}")
            logger.info(f"Test dataset size is {len(self._test_dataset)}")
            self._initialized = True

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    """
    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
        )
    """

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
        )
