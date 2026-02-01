from phonemizer.backend import EspeakBackend

from .symbols import symbols, _symbol_to_id
from .cleaners import to_lowercase, collapse_whitespace, to_ascii

global_backend = EspeakBackend(
    language='ro',
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)

def text_to_phoneme(text : str, phoemizer : EspeakBackend):
    text = to_lowercase(text)
    phonemes = phoemizer.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
    Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence
