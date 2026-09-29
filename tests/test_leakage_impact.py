"""Synthetic lexical cases only; never print real resume content."""
import pytest

from src.error_analysis import label_present, normalized
from src.leakage_impact import mask_own_label


@pytest.mark.parametrize("text,category", [
    ("Senior ACCOUNTANT; accounting and accountants.", "ACCOUNTANT"),
    ("Business-development / business___development", "BUSINESS-DEVELOPMENT"),
    ("Human resources; human resource; HR", "HR"),
    ("Digital\nmedia and digital/media", "DIGITAL-MEDIA"),
    ("ARTIST and art, artists and arts", "ARTS"),
])
def test_removes_all_audit_variants(text, category):
    assert label_present(normalized(text), category)
    assert not label_present(normalized(mask_own_label(text, category)), category)


def test_preserves_unrelated_content_and_word_boundaries():
    assert mask_own_label("parts, earth; sales!", "ARTS") == "parts, earth; sales!"
    assert mask_own_label("sales! accounting", "ACCOUNTANT") == "sales!           "


def test_casefold_offset_mapping():
    text = "Stra\u00dfe: ACCOUNTANT, caf\u00e9."
    assert mask_own_label(text, "ACCOUNTANT") == "Stra\u00dfe:           , caf\u00e9."


def test_empty_text():
    assert mask_own_label("", "HR") == ""


def test_unknown_category_rejected():
    with pytest.raises(KeyError):
        mask_own_label("synthetic", "UNKNOWN")
