<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# Grad-TTS

Official implementation of the Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

**Authors**: Vadim Popov\*, Ivan Vovk\*, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov.

<sup>\*Equal contribution.</sup>

## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## Installation

### From GitHub (pip)

```bash
pip install git+https://github.com/adrianstanea/Speech-Backbones.git#subdirectory=Grad-TTS

# Build the monotonic alignment C extension
cd $(python -c "import gradtts_ro; print(gradtts_ro.__path__[0])")/model/monotonic_align
python setup.py build_ext && cp build/lib.*/gradtts_ro/model/monotonic_align/core*.so .
```

### Local Development (uv recommended)

```bash
cd Grad-TTS

# Install the package and all dependencies
uv sync

# Build the monotonic alignment C extension
cd gradtts_ro/model/monotonic_align
uv run python setup.py build_ext && cp build/lib.*/gradtts_ro/model/monotonic_align/core*.so .
cd ../../..
```

### Local Development (pip)

```bash
cd Grad-TTS

pip install -e .

# Build the monotonic alignment C extension
cd gradtts_ro/model/monotonic_align
python3 setup.py build_ext && cp build/lib.*/gradtts_ro/model/monotonic_align/core*.so .
cd ../../..
```

> **Note:** The C extension compiles `core.pyx` into a shared library (`.so`). The build output is copied to the package directory for runtime import.

## Quick Start

### From HuggingFace Hub

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS")
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

### From local checkpoints

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_local(
    model_path="path/to/grad-tts-base-1000.pt",
    vocoder_path="path/to/hifigan_univ_v1",
    vocoder_config_path="path/to/hifigan_config.json",
)
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

### Using the example script

```bash
# From HuggingFace Hub
uv run python examples/synthesize.py

# From local checkpoints
uv run python examples/synthesize.py \
    --model-path path/to/grad-tts-base-1000.pt \
    --vocoder-path path/to/hifigan_univ_v1 \
    --vocoder-config path/to/hifigan_config.json

# Custom text
uv run python examples/synthesize.py --text "Salut, cum te cheama?"
```

## API Reference

### `GradTTSRomanian.from_pretrained()`

```python
pipe = GradTTSRomanian.from_pretrained(
    repo_id="adrianstanea/Ro-Grad-TTS",  # HuggingFace repo
    variant="base",                       # Model variant (see below)
    device=None,                          # "cpu", "cuda", or auto-detect
)
```

### `GradTTSRomanian.from_local()`

```python
pipe = GradTTSRomanian.from_local(
    model_path="path/to/grad-tts.pt",
    vocoder_path="path/to/hifigan_univ_v1",
    vocoder_config_path="path/to/hifigan_config.json",
    config_path=None,   # Optional: model config JSON, uses defaults if None
    device=None,        # "cpu", "cuda", or auto-detect
)
```

### Synthesis

```python
# Single text
audio = pipe("Romanian text here.", n_timesteps=50, temperature=1.5, length_scale=0.91)

# Batch
audios = pipe(["Text one.", "Text two."])

# Save to file
pipe.save_wav(audio, "output.wav")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_timesteps` | 50 | Diffusion reverse steps (more = higher quality, slower) |
| `temperature` | 1.5 | Sampling variance (higher = more diverse) |
| `length_scale` | 0.91 | Speech pace (> 1.0 = slower) |
| `speaker_id` | None | Speaker ID for multi-speaker models |

## Available Model Variants

| Variant | Speaker | Samples | Fine-tune Epochs |
|---------|---------|---------|-----------------|
| `base` | SWARA baseline | all | 1000 |
| `bas-10_15` | BAS | 10 | 15 |
| `bas-10_50` | BAS | 10 | 50 |
| `bas-10_100` | BAS | 10 | 100 |
| `bas-950_15` | BAS | 950 | 15 |
| `bas-950_50` | BAS | 950 | 50 |
| `bas-950_100` | BAS | 950 | 100 |
| `sgs-10_15` | SGS | 10 | 15 |
| `sgs-10_50` | SGS | 10 | 50 |
| `sgs-10_100` | SGS | 10 | 100 |
| `sgs-950_15` | SGS | 950 | 15 |
| `sgs-950_50` | SGS | 950 | 50 |
| `sgs-950_100` | SGS | 950 | 100 |

```python
# Load a specific variant
pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS", variant="bas-950_100")
```

## Model Weights

Checkpoints are hosted on HuggingFace Hub at [`adrianstanea/Ro-Grad-TTS`](https://huggingface.co/adrianstanea/Ro-Grad-TTS):

```
adrianstanea/Ro-Grad-TTS/
├── config.json
├── hifigan_config.json
├── models/
│   ├── swara/grad-tts-base-1000.pt
│   ├── bas/grad-tts-bas-{10,950}_{15,50,100}.pt
│   ├── sgs/grad-tts-sgs-{10,950}_{15,50,100}.pt
│   └── vocoder/hifigan_univ_v1
```

## Project Structure

```
Grad-TTS/
├── gradtts_ro/                  # Complete pip-installable package
│   ├── __init__.py
│   ├── pipeline.py              # GradTTSRomanian inference API
│   ├── model/                   # Grad-TTS architecture
│   │   ├── tts.py               # GradTTS module
│   │   ├── text_encoder.py      # Text encoder
│   │   ├── diffusion.py         # Diffusion decoder
│   │   └── monotonic_align/     # MAS C extension (requires build step)
│   ├── text_processing/         # Romanian phonemization
│   │   ├── __init__.py          # Espeak backend, converters
│   │   ├── symbols.py           # IPA symbol set (179 symbols)
│   │   └── cleaners.py          # Text normalization
│   └── vocoder/                 # HiFi-GAN vocoder
│       ├── models.py            # Generator architecture
│       ├── meldataset.py        # mel_spectrogram function
│       └── env.py               # AttrDict helper
├── examples/
│   └── synthesize.py            # Example synthesis script
├── train.py                     # Training script
├── data.py                      # Dataset classes
├── finetune.sh                  # Speaker fine-tuning script
├── pyproject.toml               # Package definition
└── uv.lock                      # Reproducible dependency lock
```

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. For single speaker training refer to `jspeech` filelists and to `libri-tts` filelists for multispeaker.
2. Set experiment configuration in `params.py` file.
3. Specify your GPU device and run training script:
    ```bash
    export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
    python train.py  # if single speaker
    python train_multi_speaker.py  # if multispeaker
    ```
4. To track your training process run tensorboard server on any available port:
    ```bash
    tensorboard --logdir=YOUR_LOG_DIR --port=8888
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).
