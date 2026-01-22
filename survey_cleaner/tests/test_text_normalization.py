"""Tests for text normalization functions."""

import pytest
from core.text_normalization import (
    fix_typos_and_abbreviations,
    is_postdoc_title,
    normalize_title
)

def test_fix_typos_and_abbreviations():
    """Test typo and abbreviation correction."""
    assert fix_typos_and_abbreviations("Asst. Professor") == "assistant professor"
    assert fix_typos_and_abbreviations("Assoc. Prof.") == "associate professor"
    assert fix_typos_and_abbreviations("Visiting Asst. Prof") == "visiting assistant professor"

def test_is_postdoc_title():
    """Test postdoctoral title detection."""
    assert is_postdoc_title("Postdoctoral Fellow")
    assert is_postdoc_title("Post-doc Fellow")
    assert is_postdoc_title("Postdoc Researcher")
    assert not is_postdoc_title("Assistant Professor")

def test_normalize_title():
    """Test title normalization."""
    from config.settings import JOB_KEYWORDS
    
    result = normalize_title("Senior Asst. Professor of Biology", JOB_KEYWORDS)
    assert "assistant professor" in result
    
    result = normalize_title("Postdoctoral Research Fellow", JOB_KEYWORDS)
    assert result == "postdoctoral"
