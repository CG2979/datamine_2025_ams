"""Text normalization and typo correction utilities."""

import re
from typing import Dict
from config.settings import (
    ABBREVIATION_REPLACEMENTS, 
    POSTDOC_REPLACEMENTS, 
    COMMON_TYPO_REPLACEMENTS
)

def get_all_replacements() -> Dict[str, str]:
    """Combine all replacement dictionaries."""
    return {
        **ABBREVIATION_REPLACEMENTS,
        **POSTDOC_REPLACEMENTS,
        **COMMON_TYPO_REPLACEMENTS
    }

def fix_typos_and_abbreviations(title: str) -> str:
    """Fix common typos and abbreviations in a title."""
    t = title.lower()
    
    for pattern, replacement in get_all_replacements().items():
        t = re.sub(pattern, replacement, t)
    
    # Remove duplicate consecutive words
    words = t.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i-1]:
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words)

def is_postdoc_title(title: str) -> bool:
    """Check if a title is postdoctoral-related."""
    t = str(title).lower()
    return (
        'postdoctoral' in t or
        'postdoc fellow' in t or
        'post-doc fellow' in t or
        'post doctoral fellow' in t or
        'post doc' in t or
        'postdoctor' in t
    )

def normalize_title(title: str, job_keywords: list) -> str:
    """Normalize a job title to its canonical form."""
    # Fix typos and abbreviations
    t = fix_typos_and_abbreviations(title)
    
    # Remove punctuation
    t = re.sub(r"[^a-z0-9\s]", "", t)
    
    # Remove extra spaces
    t = re.sub(r"\s+", " ", t).strip()
    
    # Special handling for postdoctoral
    if re.search(r"\bpostdoctoral\b", t):
        return "postdoctoral"
    
    # Modifiers to preserve
    PRESERVE_MODIFIERS = ["associate", "assistant"]
    
    # Find longest matching keyword
    for keyword in sorted(job_keywords, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(keyword)}\b", t)
        if match:
            start_idx = match.start()
            prefix = t[:start_idx].strip()
            words_before = prefix.split()
            
            if words_before and words_before[-1] in PRESERVE_MODIFIERS:
                modifier_start = t.rfind(words_before[-1], 0, start_idx)
                return t[modifier_start:].strip()
            else:
                return t[start_idx:].strip()
    
    return t
