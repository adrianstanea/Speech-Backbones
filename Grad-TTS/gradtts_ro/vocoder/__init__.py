from .env import AttrDict
from .models import Generator as HiFiGAN


def get_mel_spectrogram():
    """Lazy import of mel_spectrogram to avoid librosa dependency at import time."""
    from .meldataset import mel_spectrogram
    return mel_spectrogram


__all__ = ["AttrDict", "HiFiGAN", "get_mel_spectrogram"]
