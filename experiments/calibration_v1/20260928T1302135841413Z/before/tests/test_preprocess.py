import pytest
from src.preprocess import clean_text


@pytest.mark.parametrize(
    "text", ["", "   ", "!!!@@@ 12345", "the and of", "https://example.com/skills"]
)
def test_empty_after_cleaning(text):
    assert clean_text(text) == ""


def test_normalization():
    assert (
        clean_text("The SKILLS, systems & databases! https://example.com")
        == "skill system database"
    )


@pytest.mark.parametrize("text", [None, 42, ["python"]])
def test_non_string(text):
    with pytest.raises(TypeError):
        clean_text(text)


def test_repeatable():
    text = "Managed databases and teams."
    assert clean_text(clean_text(text)) == clean_text(text)
