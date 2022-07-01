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
    def __init__(self, waveform: Tensor, sample_rate):
        self._waveform = waveform
        self._

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
