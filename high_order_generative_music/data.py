import torchaudio
from torch.utils.data import Dataset
from torch import Tensor


def single_recording_dataset(filename: str):
    with open(filename, "rb") as file:

        metadata = torchaudio.info(filename)
        waveform, sample_rate = torchaudio.load(file)
        return waveform, sample_rate


class SingleRecordingDataset(Dataset):
    def __init__(
        self,
        waveform: Tensor,
        sample_rate,
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
        return (
            self._waveform[idx:offset],
            self._waveform[offset : (offset + self._output_window_size)],
        )
