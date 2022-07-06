import torchaudio
from torch.utils.data import Dataset
from torch import Tensor


def single_recording_dataset(filename: str):
    with open(filename, "rb") as file:

        metadata = torchaudio.info(filename)
        print(metadata)
        waveform, sample_rate = torchaudio.load(file)

        print("sample_rate", sample_rate)
        print("waveform", waveform)
        print("waveform.shape", waveform.shape)
        return waveform, sample_rate


class SingleRecordingDataset(Dataset):
    def __init__(
        self, waveform: Tensor, sample_rate, window_size: int = 1000, channel: int = 0
    ):
        self._waveform = waveform[channel]
        self._sample_rate = sample_rate
        self._window_size = window_size
        self._size = self._waveform.shape[0] - (window_size + 1)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):

        return (
            self._waveform[idx : (idx + self._window_size)],
            self._waveform[idx : (idx + self._window_size + 1)],
        )
