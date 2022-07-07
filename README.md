[![CI](https://github.com/jloveric/high-order-generative-music/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/high-order-generative-music/actions/workflows/python-app.yml)
# high-order-generative-music
Experiments in generative music using [high order layers](https://github.com/jloveric/high-order-layers-torch).
Generated audio and waveforms can be seen in the tensorboard output.

[Free music archive](https://freemusicarchive.org/) tool is located [here](https://github.com/mdeff/fma) and the 

# Installation
Make sure to install ffmpeg as it's used by torchaudio.  When running things on the command line
```
poetry install
```
and then
```
poetry shell
```
# Examples
Try and memorize as set
```
python examples/memorize.py
```
to try with shorter versions
```
python examples/memorize.py data.max_size=100000
```

# Relevant Papers
[Goodbye Wavenet](https://syncedreview.com/2022/06/22/a-wavenet-rival-stanford-u-study-models-raw-audio-waveforms-over-contexts-of-500k-samples/)

[It's Raw: Audio Generation with State Space Models](https://arxiv.org/pdf/2202.09729.pdf)

[Wavenet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)