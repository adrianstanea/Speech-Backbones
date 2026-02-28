# Grad-TTS for Romanian (Ro-Grad-TTS)

Adaptation of [Grad-TTS](https://arxiv.org/abs/2105.06337) for Romanian text-to-speech synthesis, trained on the SWARA 1.0 dataset.

## Prerequisites

- Python >= 3.9
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (system package, required for Romanian phonemization)
- A C compiler (gcc/clang) for building the Cython extension

Install espeak-ng:

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng
```

## Installation

### Using uv (recommended)

```bash
cd Speech-Backbones/Grad-TTS

# Install the package and all dependencies in a virtual environment
uv sync

# Build the monotonic alignment C extension
cd model/monotonic_align
uv run python setup.py build_ext --inplace
cd ../..
```

### Using pip

```bash
cd Speech-Backbones/Grad-TTS

# Make sure to use a Python 3.9 interpreter
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Build the monotonic alignment C extension (must run from this directory)
cd model/monotonic_align
python3 setup.py build_ext --inplace
cd ../..
```

> **Note:** The `setup.py` for monotonic_align must be run from inside `model/monotonic_align/`, not from the project root. It compiles `core.pyx` into a shared library (`.so`) used at runtime.

## Quick Start

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

### Synthesis with the `GradTTSRomanian` pipeline:

```python
# Load a specific variant
pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS", variant="bas-950_100")

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

```python
```

## Training

Training requires the optional `[train]` dependencies and the SWARA 1.0 dataset.

```bash
uv sync --extra train
```

### Base Model

Edit `params.py` to point to your SWARA 1.0 filelists, then:

```bash
uv run python train.py
```

### Fine-tuning for Specific Speakers

```bash
uv run python train.py \
    --checkpoint_path path/to/base_model.pt \
    --train_filelist_path resources/filelists/SWARA_1_0/finetune_meta_bas_10.csv \
    --n_epochs 100 \
    --batch_size 5 \
    --log_dir log_dir/bas_10
```

Or use the batch fine-tuning script (edit paths inside first):

```bash
./finetune.sh
```

## Key Changes from Original Grad-TTS

- **Phonemization**: Espeak (Romanian) replaces CMUDict (English)
- **Model**: `spk_emb_dim` and `n_spks` passed to TextEncoder; mel cropping fix
- **Training**: Argparse support, checkpoint resume, SWARA 1.0 integration
- **Inference**: `gradtts_ro` package with HuggingFace Hub integration, CPU/CUDA auto-detection

## Citation


If you use this Romanian adaptation in your research, please cite:

```bibtex
@ARTICLE{11269795,
  author={Răgman, Teodora and Bogdan Stânea, Adrian and Cucu, Horia and Stan, Adriana},
  journal={IEEE Access},
  title={How Open Is Open TTS? A Practical Evaluation of Open Source TTS Tools},
  year={2025},
  volume={13},
  number={},
  pages={203415-203428},
  keywords={Computer architecture;Training;Text to speech;Spectrogram;Decoding;Computational modeling;Codecs;Predictive models;Acoustics;Low latency communication;Speech synthesis;open tools;evaluation;computational requirements;TTS adaptation;text-to-speech;objective measures;listening test;Romanian},
  doi={10.1109/ACCESS.2025.3637322}
}
```


```bibtex
@article{popov2021grad,
  title={Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech},
  author={Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tasnima and Kudinov, Mikhail},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).

## License

MIT License. See [LICENSE](LICENSE) for details.
