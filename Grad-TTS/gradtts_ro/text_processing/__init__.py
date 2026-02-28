"""
Text processing for Romanian TTS.

Provides phonemization using espeak-ng and text cleaning utilities.
"""

from .symbols import symbols, _symbol_to_id
from .cleaners import to_lowercase, collapse_whitespace, to_ascii

# Lazy-initialized global phonemizer backend for training compatibility
_global_backend = None


def create_phonemizer():
    """Create an Espeak backend for Romanian phonemization."""
    from phonemizer.backend import EspeakBackend
    return EspeakBackend(
        language='ro',
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )


def _get_global_backend():
    """Get or create the global phonemizer backend (lazy initialization)."""
    global _global_backend
    if _global_backend is None:
        _global_backend = create_phonemizer()
    return _global_backend


# For backwards compatibility with training code that uses `global_backend` directly
class _LazyBackend:
    """Proxy that lazily initializes the phonemizer backend on first use."""

    def __getattr__(self, name):
        return getattr(_get_global_backend(), name)

    def phonemize(self, *args, **kwargs):
        return _get_global_backend().phonemize(*args, **kwargs)


global_backend = _LazyBackend()


def text_to_phoneme(text, phonemizer=None):
    """Convert text to phoneme string using the given phonemizer backend.

    Args:
        text: Input text string.
        phonemizer: Phonemizer backend. If None, uses global_backend.

    Returns:
        Phoneme string.
    """
    if phonemizer is None:
        phonemizer = _get_global_backend()
    text = to_lowercase(text)
    phonemes = phonemizer.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of phonemes to a sequence of IDs corresponding to the symbols.

    Args:
        cleaned_text: String of phonemes to convert.

    Returns:
        List of integers corresponding to the symbols in the text.
    """
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def intersperse(lst, item):
    """Adds blank symbol between each element.

    Args:
        lst: Input list.
        item: Item to insert between elements.

    Returns:
        New list with item interspersed.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


__all__ = [
    "symbols",
    "_symbol_to_id",
    "to_lowercase",
    "collapse_whitespace",
    "to_ascii",
    "create_phonemizer",
    "global_backend",
    "text_to_phoneme",
    "cleaned_text_to_sequence",
    "intersperse",
]
