"""Experimental features with the production universal mask preserved."""
from functools import lru_cache
from src.label_masking import universal_mask,strip_phrases
from src.preprocess import masked_feature_phrases
from src.v3_features import clean_technical


@lru_cache(maxsize=4096)
def masked_technical(text: str) -> str:
    """Retain technical spellings, never restore excluded category phrases.

    Character features receive this SAME masked string, not the unmasked source.
    Caches are in RAM only; no raw resume text is written or logged.
    """
    return ' '.join(strip_phrases(clean_technical(universal_mask(text)),
                                 masked_feature_phrases()).split())
