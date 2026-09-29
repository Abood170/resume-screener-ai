"""Experimental inference-safe features; does not alter production preprocessing.

Retaining capped body label words does NOT eliminate the label shortcut. A cap
limits repetition, not the information in presence/absence of a category word.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from src.label_masking import VARIANTS
from src.preprocess import clean_text_legacy

# Protect technical shapes before alphabetic-only cleaning. URLs are removed
# first. Generic hyphenated words can also be retained: an explicit limitation.
TECH = re.compile(r"(?<![\w])(?:\.[a-zA-Z][\w]*|[a-zA-Z][\w]*(?:\+{1,2}|#)|[a-zA-Z][\w]*(?:[./-][a-zA-Z0-9][\w]*)+|[a-zA-Z]+[0-9]+[\w]*)(?![\w])")


@lru_cache(maxsize=4096)
def clean_technical(text: str) -> str:
    """Placeholder-and-restore; no document text is persisted or logged."""
    if not isinstance(text, str): raise TypeError('text must be a string')
    text=re.sub(r'https?://\S+|www\.\S+', ' ', text)
    prefix='zztechnicalplaceholder'
    while prefix in text.lower(): prefix+='z'
    restored={}
    def protect(match):
        # Alphabetic suffix survives the legacy tokenizer and noun lemmatizer.
        n=len(restored); suffix=''
        while True:
            suffix=chr(97+n%26)+suffix; n=n//26-1
            if n<0: break
        key=prefix+suffix+'zz'
        restored[key]=match.group().lower()
        return ' '+key+' '
    protected=TECH.sub(protect,text)
    return ' '.join(restored.get(t,t) for t in clean_text_legacy(protected).split())


@lru_cache(maxsize=1)
def variant_tokens():
    return {category:tuple(sorted({tuple(clean_technical(term).split())
            for term in [category,*variants]},key=lambda t:(-len(t),t)))
            for category,variants in VARIANTS.items()}


def occurrences(tokens, phrases):
    """Unique token spans, longest matching phrase at each start position."""
    matches=[]
    for start in range(len(tokens)):
        for phrase in phrases:
            if phrase and tuple(tokens[start:start+len(phrase)])==phrase:
                matches.append((start,start+len(phrase))); break
    return matches


@dataclass(frozen=True)
class SmartMasker:
    k: int

    @lru_cache(maxsize=4096)
    def __call__(self, text: str) -> str:
        tokens=clean_technical(text).split()
        phrases=sorted({p for items in variant_tokens().values() for p in items},key=lambda t:(-len(t),t))
        remove=set()
        # A phrase starting in the prefix is removed in full even if it crosses K.
        for start,end in occurrences(tokens,phrases):
            if start<self.k: remove.update(range(start,end))
        return ' '.join(t for i,t in enumerate(tokens) if i not in remove)


@lru_cache(maxsize=1)
def capped_tokens():
    # Cap unigram components too, including multiword labels longer than the
    # vectorizer's bigram window. This also caps common words such as 'business'.
    return frozenset(t for items in variant_tokens().values() for phrase in items for t in phrase)


class CappedTfidfVectorizer(TfidfVectorizer):
    """Clip raw counts of label-containing unigram/bigram features before TF-IDF.

    Body text is retained. Its matching features contribute at most `cap` counts
    per feature, before sublinear TF, IDF and document normalization. This does
    not impose a collective budget across different label-containing bigrams.
    """
    def __init__(self, k=20, cap=2, min_df=2, max_df=0.95,
                 max_features=5000, sublinear_tf=True):
        self.k=k
        self.cap=cap
        super().__init__(preprocessor=SmartMasker(k),tokenizer=str.split,
            token_pattern=None,lowercase=False,ngram_range=(1,2),
            min_df=min_df,max_df=max_df,max_features=max_features,sublinear_tf=sublinear_tf)

    def build_analyzer(self):
        # Rebuild from estimator parameters after Grid/RandomizedSearch set_params.
        self.preprocessor=SmartMasker(self.k)
        analyze=super().build_analyzer()
        limited=capped_tokens()
        cap=self.cap
        def analyzer(document):
            counts=Counter()
            for feature in analyze(document):
                if any(t in limited for t in feature.split()):
                    counts[feature]+=1
                    if counts[feature]>cap: continue
                yield feature
        return analyzer
