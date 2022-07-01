import mlflow
import hydra
from hydra import utils
from omegaconf import DictConfig
from high_order_generative_music.plotting import plot_waveform, plot_specgram
from high_order_generative_music.data import single_recording_dataset


@hydra.main(config_path="../configs", config_name="memorize")
def memorize(cfg: DictConfig):

    base_dir = utils.get_original_cwd()
    mlflow.set_tracking_uri(f"file://{base_dir}/mlruns")

    filename = f"{base_dir}/music/TeaKPea-vpunk.mp3"
    waveform, sample_rate = single_recording_dataset(filename=filename)
    print("waveform.shape", waveform[0].shape, waveform[1].shape)
    plot_waveform(waveform=waveform, sample_rate=sample_rate, window=list(range(10000)))


if __name__ == "__main__":
    memorize()
