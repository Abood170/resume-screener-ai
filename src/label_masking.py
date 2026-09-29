"""Fixed, label-independent vocabulary masking shared by fitting and inference.

The original hypothesis is that performance might fall between baseline_v1 and
L1. This is NOT a bound: universal masking removes more content than L1 and can
perform worse. Measure the result; never assume that L1 predicts this score.
The dictionary is copied verbatim from the earlier audit, not learned from data.
"""
from functools import lru_cache
import re

# Kept local so inference does not import diagnostic plotting/data dependencies.
VARIANTS = {
    "ACCOUNTANT": ["accountant", "accountants", "accounting"],
    "ADVOCATE": ["advocate", "advocates", "advocacy"],
    "AGRICULTURE": ["agriculture", "agricultural"],
    "APPAREL": ["apparel"],
    "ARTS": ["arts", "artist", "artists", "art"],
    "AUTOMOBILE": ["automobile", "automobiles", "automotive"],
    "AVIATION": ["aviation"],
    "BANKING": ["banking", "banker", "bankers"],
    "BPO": ["bpo", "business process outsourcing"],
    "BUSINESS-DEVELOPMENT": ["business development"],
    "CHEF": ["chef", "chefs"],
    "CONSTRUCTION": ["construction"],
    "CONSULTANT": ["consultant", "consultants", "consulting", "consultancy"],
    "DESIGNER": ["designer", "designers", "design"],
    "DIGITAL-MEDIA": ["digital media"],
    "ENGINEERING": ["engineering", "engineer", "engineers"],
    "FINANCE": ["finance", "financial"],
    "FITNESS": ["fitness"],
    "HEALTHCARE": ["healthcare", "health care"],
    "HR": ["hr", "human resources", "human resource"],
    "INFORMATION-TECHNOLOGY": ["information technology"],
    "PUBLIC-RELATIONS": ["public relations"],
    "SALES": ["sales", "salesperson"],
    "TEACHER": ["teacher", "teachers", "teaching"],
}
PHRASES = tuple(sorted({re.sub(r"[\W_]+", " ", v.casefold()).strip()
                        for category, variants in VARIANTS.items()
                        for v in [category, *variants]}))


@lru_cache(maxsize=4)
def _pattern(phrases: tuple[str, ...]) -> re.Pattern:
    choices = [r"[\W_]+".join(map(re.escape, phrase.split()))
               for phrase in sorted(phrases, key=len, reverse=True) if phrase]
    # Underscore and punctuation separate words, as in the original audit.
    return re.compile(r"(?<![^\W_])(?:" + "|".join(choices) + r")(?![^\W_])", re.I)


def strip_phrases(text: str, phrases: tuple[str, ...]) -> str:
    """Remove all fixed phrases, including matches exposed by earlier removals."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    pattern = _pattern(phrases)
    while True:
        updated, matches = pattern.subn(" ", text)
        if not matches:
            return updated
        text = updated


def universal_mask(text: str) -> str:
    """No category argument: every input uses the same complete dictionary."""
    return strip_phrases(text, PHRASES)
