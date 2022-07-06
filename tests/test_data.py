import pytest
from high_order_generative_music.data import (
    single_recording_dataset,
    SingleRecordingDataset,
)
from torch.utils.data import DataLoader


def test_single_recording_dataset():
    filename = "music/TeaKPea-vpunk.mp3"

    waveform, sample_rate = single_recording_dataset(filename=filename)

    dataset = SingleRecordingDataset(
        waveform=waveform, sample_rate=sample_rate, window_size=100, channel=0
    )

    dataloader = DataLoader(dataset=dataset, batch_size=10)
    dataiter = iter(dataloader)

    features, targets = dataiter.next()

    print("features", features, "targets", targets)
