"""Deterministic preprocessing shared by training and inference.

Install corpora explicitly: python -m nltk.downloader stopwords wordnet omw-1.4
No network requests are made during inference.
"""

from functools import lru_cache
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@lru_cache(maxsize=1)
def resources() -> tuple[frozenset[str], WordNetLemmatizer]:
    words = frozenset(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("skills")  # Fail early if wordnet is missing.
    return words, lemmatizer


def clean_text_legacy(text: str) -> str:
    """Lowercase, remove URLs/nonletters, tokenize, filter and noun-lemmatize.

    Alphabetic tokenization deliberately loses distinctions like C++ versus C#.
    Noun lemmatization is conservative; no POS tagger is implied.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    words, lemmatizer = resources()
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(
        lemmatizer.lemmatize(t) for t in tokens if t not in words and len(t) > 1
    )


def clean_text(text: str) -> str:
    """Legacy pickle-compatible entry point; frozen vectorizers reference this.

    Do not change its semantics: baseline and L1 must remain reproducible.
    Production v2 vectorizers explicitly bind clean_text_universal instead.
    """
    return clean_text_legacy(text)


@lru_cache(maxsize=1)
def masked_feature_phrases() -> tuple[str, ...]:
    from src.label_masking import PHRASES
    return tuple(sorted({clean_text_legacy(phrase) for phrase in PHRASES}))


def clean_text_universal(text: str) -> str:
    """Production preprocessing, identical during CV, fitting, and inference.

    Mask before cleaning, then remove canonical lemma forms so lemmatization or
    removed stopwords cannot reintroduce excluded phrases into the vocabulary.
    This also suppresses singular 'sale' (the lemma of 'sales'). It removes
    genuine domain evidence too; it is not a guarantee against all shortcuts.
    """
    from src.label_masking import strip_phrases, universal_mask
    cleaned = clean_text_legacy(universal_mask(text))
    return " ".join(strip_phrases(cleaned, masked_feature_phrases()).split())
