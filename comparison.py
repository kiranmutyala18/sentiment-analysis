"""Optional DistilBERT benchmark against the local TF-IDF baseline."""

from __future__ import annotations

from typing import Iterable


def classify_with_distilbert(texts: Iterable[str]) -> list[dict[str, object]]:
    """Download and run a small public transformer only when explicitly invoked."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("Install the optional [transformer] dependencies first.") from error
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return list(classifier(list(texts)))
