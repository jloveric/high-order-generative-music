import pytest
from high_order_generative_music.data import single_recording_dataset


def test_single_recording_dataset():
    filename = "music/Tea K Pea - vpunk.mp3"

    single_recording_dataset(filename=filename)
