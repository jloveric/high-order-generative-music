[![CI](https://github.com/jloveric/high-order-generative-music/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/high-order-generative-music/actions/workflows/python-app.yml)
# high-order-generative-music
Experiments in generative music using [high order layers](https://github.com/jloveric/high-order-layers-torch).
Generated audio and waveforms can be seen in the tensorboard output. This is a work in progress and I don't currently have
any good results.

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
Try and memorize as set using conv net
```
python examples/memorize.py
```
with a tail focus network
```
python examples/memorize.py net=tail_focus
```
to try with shorter versions
```
python examples/memorize.py data.max_size=100000
```
and for debugging
```
python examples/memorize.py data.max_size=10000 data.window_size=1000
```
run mlflow
```
poetry run mlflow ui
```

# Relevant Papers
[Goodbye Wavenet](https://syncedreview.com/2022/06/22/a-wavenet-rival-stanford-u-study-models-raw-audio-waveforms-over-contexts-of-500k-samples/)

[It's Raw: Audio Generation with State Space Models](https://arxiv.org/pdf/2202.09729.pdf)

[Wavenet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)