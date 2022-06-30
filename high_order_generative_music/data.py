import torchaudio


def single_recording_dataset(filename: str):
    with open(filename, "rb") as file:
        waveform, sample_rate = torchaudio.load(file)
