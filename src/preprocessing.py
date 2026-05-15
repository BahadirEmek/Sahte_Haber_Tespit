"""Text preprocessing utilities for the fake news detection project."""

from __future__ import annotations

import re
import string
from functools import lru_cache

try:
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover - dependency guard
    stopwords = None


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return English stopwords with a safe fallback when NLTK data is missing."""
    if stopwords is None:
        return FALLBACK_STOPWORDS

    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


class TextPreprocessor:
    """Clean raw news text before model training or prediction."""

    def __init__(self, stop_words: set[str] | None = None) -> None:
        """Create a reusable preprocessor with compiled cleanup helpers."""
        self.stop_words = stop_words if stop_words is not None else get_stopwords()
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.html_pattern = re.compile(r"<.*?>")
        self.digit_pattern = re.compile(r"\d+")
        self.space_pattern = re.compile(r"\s+")
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: str) -> str:
        """Normalize and clean one text value for the classifier.

        The method handles ``None`` and blank values, lowercases text,
        removes URLs, HTML tags, punctuation, digits, extra spaces, and
        English stopwords.
        """
        if text is None:
            return ""

        normalized_text = str(text).strip().lower()
        if not normalized_text:
            return ""

        normalized_text = self.url_pattern.sub(" ", normalized_text)
        normalized_text = self.html_pattern.sub(" ", normalized_text)
        normalized_text = normalized_text.translate(self.punctuation_table)
        normalized_text = self.digit_pattern.sub(" ", normalized_text)
        normalized_text = self.space_pattern.sub(" ", normalized_text).strip()

        if not normalized_text:
            return ""

        return " ".join(
            word for word in normalized_text.split() if word not in self.stop_words
        )


_DEFAULT_PREPROCESSOR = TextPreprocessor()


def clean_text(text: str) -> str:
    """Clean raw news text using the default ``TextPreprocessor`` instance."""
    return _DEFAULT_PREPROCESSOR.clean_text(text)
