"""Optional LIME explanation for a persisted baseline model."""

from __future__ import annotations

from model import SentimentModel


def explain_with_lime(model: SentimentModel, text: str) -> list[tuple[str, float]]:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as error:
        raise RuntimeError("Install the optional [explain] dependencies first.") from error
    pipeline = model._pipeline
    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
    return [(word, float(weight)) for word, weight in explanation.as_list()]
