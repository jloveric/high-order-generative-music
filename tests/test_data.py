import pytest
from high_order_generative_music.data import (
    single_recording_dataset,
    SingleRecordingDataset,
    SingleRecordingDataModule,
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

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == 100
    assert targets.shape[1] == 1


def test_random_image_sample_datamodule():
    filename = "music/TeaKPea-vpunk.mp3"

    dataset = SingleRecordingDataModule(
        filename=filename,
        window_size=1000,
        output_window_size=1,
        batch_size=32,
        num_workers=1,
    )

    dataset.setup()

    assert len(dataset.train_dataset) > 0

    dataloader = dataset.train_dataloader()

    features, targets = iter(dataloader).next()

    assert features.shape[1] == 1000
    assert targets.shape[1] == 1
    assert features.shape[0] == targets.shape[0]
