import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from gradtts_ro.text_processing import (
    create_phonemizer,
    text_to_phoneme,
    cleaned_text_to_sequence,
    intersperse,
    symbols,
)
from gradtts_ro.vocoder import HiFiGAN, AttrDict
from gradtts_ro.model import GradTTS


# Default model hyperparameters
DEFAULT_CONFIG = {
    "n_spks": 1,
    "spk_emb_dim": 64,
    "n_enc_channels": 192,
    "filter_channels": 768,
    "filter_channels_dp": 256,
    "n_heads": 2,
    "n_enc_layers": 6,
    "enc_kernel": 3,
    "enc_dropout": 0.1,
    "window_size": 4,
    "n_feats": 80,
    "dec_dim": 64,
    "beta_min": 0.05,
    "beta_max": 20.0,
    "pe_scale": 1000,
    "sample_rate": 22050,
    "add_blank": True,
}

# Variant name -> (subfolder, filename) - in the HuggingFace repository
VARIANT_MAP = {
    "base": ("models/swara", "grad-tts-base-1000.pt"),
    "bas-10_15": ("models/bas", "grad-tts-bas-10_15.pt"),
    "bas-10_50": ("models/bas", "grad-tts-bas-10_50.pt"),
    "bas-10_100": ("models/bas", "grad-tts-bas-10_100.pt"),
    "bas-950_15": ("models/bas", "grad-tts-bas-950_15.pt"),
    "bas-950_50": ("models/bas", "grad-tts-bas-950_50.pt"),
    "bas-950_100": ("models/bas", "grad-tts-bas-950_100.pt"),
    "sgs-10_15": ("models/sgs", "grad-tts-sgs-10_15.pt"),
    "sgs-10_50": ("models/sgs", "grad-tts-sgs-10_50.pt"),
    "sgs-10_100": ("models/sgs", "grad-tts-sgs-10_100.pt"),
    "sgs-950_15": ("models/sgs", "grad-tts-sgs-950_15.pt"),
    "sgs-950_50": ("models/sgs", "grad-tts-sgs-950_50.pt"),
    "sgs-950_100": ("models/sgs", "grad-tts-sgs-950_100.pt"),
}

HF_VOCODER_SUBFOLDER = "models/vocoder"
HF_VOCODER_FILENAME = "hifigan_univ_v1"


class GradTTSRomanian:
    """Romanian Grad-TTS text-to-speech pipeline.

    Usage:
        # From HuggingFace Hub
        pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS")

        # From local files
        pipe = GradTTSRomanian.from_local(
            model_path="checkpts/grad-tts-ro.pt",
            vocoder_path="checkpts/hifigan_univ_v1",
            vocoder_config_path="checkpts/hifigan-config.json",
        )

        # Synthesize speech
        audio = pipe("Salut!")
        pipe.save_wav(audio, "output.wav")
    """

    def __init__(self, generator, vocoder, phonemizer, config, device):
        self.generator = generator
        self.vocoder = vocoder
        self.phonemizer = phonemizer
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        repo_id="adrianstanea/Ro-Grad-TTS",
        variant="base",
        device=None,
        revision=None,
        cache_dir=None,
    ):
        """Load model from a HuggingFace Hub repository.

        Args:
            repo_id: HuggingFace repository ID.
            variant: Model variant to load. One of:
                "base", "bas-10_100", "bas-950_100", "sgs-10_100", "sgs-950_100".
            device: Device to load model on. Auto-detects if None.
            revision: Git revision (branch/tag/commit) on the HF repo.
            cache_dir: Where to cache downloaded files.
        """
        from huggingface_hub import hf_hub_download

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Download config
        config_path = hf_hub_download(
            repo_id, "config.json",
            revision=revision, cache_dir=cache_dir,
        )
        with open(config_path) as f:
            config = json.load(f)

        # Download model checkpoint
        if variant not in VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(VARIANT_MAP.keys())}"
            )
        subfolder, filename = VARIANT_MAP[variant]
        model_path = hf_hub_download(
            repo_id, filename, subfolder=subfolder,
            revision=revision, cache_dir=cache_dir,
        )

        # Download vocoder
        vocoder_config_path = hf_hub_download(
            repo_id, "hifigan_config.json",
            revision=revision, cache_dir=cache_dir,
        )
        vocoder_path = hf_hub_download(
            repo_id, HF_VOCODER_FILENAME, subfolder=HF_VOCODER_SUBFOLDER,
            revision=revision, cache_dir=cache_dir,
        )

        return cls._build(config, model_path, vocoder_path, vocoder_config_path, device)

    @classmethod
    def from_local(
        cls,
        model_path,
        vocoder_path,
        vocoder_config_path,
        config_path=None,
        device=None,
    ):
        """Load model from local file paths.

        Args:
            model_path: Path to Grad-TTS checkpoint (.pt file).
            vocoder_path: Path to HiFi-GAN vocoder checkpoint.
            vocoder_config_path: Path to HiFi-GAN config JSON.
            config_path: Path to model config JSON. Uses defaults if None.
            device: Device to load model on. Auto-detects if None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = dict(DEFAULT_CONFIG)

        return cls._build(config, model_path, vocoder_path, vocoder_config_path, device)

    @classmethod
    def _build(cls, config, model_path, vocoder_path, vocoder_config_path, device):
        """Internal builder: constructs models from config and paths."""
        # Build Grad-TTS generator
        n_vocab = len(symbols) + 1 if config.get("add_blank", True) else len(symbols)
        generator = GradTTS(
            n_vocab,
            config["n_spks"],
            config.get("spk_emb_dim"),
            config["n_enc_channels"],
            config["filter_channels"],
            config["filter_channels_dp"],
            config["n_heads"],
            config["n_enc_layers"],
            config["enc_kernel"],
            config["enc_dropout"],
            config["window_size"],
            config["n_feats"],
            config["dec_dim"],
            config["beta_min"],
            config["beta_max"],
            config["pe_scale"],
        )
        generator.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        generator = generator.to(device).eval()

        # Build HiFi-GAN vocoder
        with open(vocoder_config_path) as f:
            h = AttrDict(json.load(f))
        vocoder = HiFiGAN(h)
        vocoder_state = torch.load(vocoder_path, map_location=device)
        if "generator" in vocoder_state:
            vocoder_state = vocoder_state["generator"]
        vocoder.load_state_dict(vocoder_state)
        vocoder = vocoder.to(device).eval()
        vocoder.remove_weight_norm()

        # Create phonemizer
        phonemizer = create_phonemizer()

        return cls(generator, vocoder, phonemizer, config, device)

    def __call__(
        self,
        text,
        n_timesteps=50,
        temperature=1.5,
        length_scale=0.91,
        speaker_id=None,
    ):
        """Synthesize speech from text.

        Args:
            text: A string or list of strings to synthesize.
            n_timesteps: Number of diffusion reverse steps.
            temperature: Controls variance of terminal distribution.
            length_scale: Controls speech pace (>1 = slower).
            speaker_id: Speaker ID for multi-speaker models (leave None, not used in the Romanian adaptation).

        Returns:
            A numpy int16 array (single text) or list of arrays (batch).
        """
        if isinstance(text, str):
            return self._synthesize_one(text, n_timesteps, temperature, length_scale, speaker_id)
        return [
            self._synthesize_one(t, n_timesteps, temperature, length_scale, speaker_id)
            for t in text
        ]

    def _synthesize_one(self, text, n_timesteps, temperature, length_scale, speaker_id):
        """Synthesize a single utterance."""
        # Text to phoneme sequence
        phonemes = text_to_phoneme(text, self.phonemizer)
        seq = cleaned_text_to_sequence(phonemes)
        if self.config.get("add_blank", True):
            seq = intersperse(seq, len(symbols))

        x = torch.LongTensor(seq).to(self.device).unsqueeze(0)
        x_lengths = torch.LongTensor([x.shape[-1]]).to(self.device)

        spk = None
        if speaker_id is not None:
            spk = torch.LongTensor([speaker_id]).to(self.device)

        with torch.no_grad():
            _, y_dec, _ = self.generator(
                x, x_lengths,
                n_timesteps=n_timesteps,
                temperature=temperature,
                stoc=False,
                spk=spk,
                length_scale=length_scale,
            )
            audio = self.vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy()

        return (audio * 32768).astype(np.int16)

    @property
    def sample_rate(self):
        return self.config.get("sample_rate", 22050)

    def save_wav(self, audio, path):
        """Save audio array to a WAV file.

        Args:
            audio: numpy int16 array from __call__.
            path: Output file path.
        """
        write_wav(path, self.sample_rate, audio)
