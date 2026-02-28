""" from https://github.com/gmltmd789/UnitSpeech.git """

import re
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def to_lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def to_ascii(text):
    return unidecode(text)