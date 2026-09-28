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


def clean_text(text: str) -> str:
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
