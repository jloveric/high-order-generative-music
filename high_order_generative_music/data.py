import torchaudio


def single_recording_dataset(filename: str):
    with open(filename, "rb") as file:

        metadata = torchaudio.info(filename)
        print(metadata)
        waveform, sample_rate = torchaudio.load(file)

        print("sample_rate", sample_rate)
        print("waveform", waveform)
        print("waveform.shape", waveform.shape)
