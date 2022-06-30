import torchaudio


def single_recording_dataset(filename: str):
    with open(filename) as file:
        waveform, sample_rate = torchaudio.load(file)
