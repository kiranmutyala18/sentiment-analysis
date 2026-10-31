"""Train, save, load, and explain a product-review sentiment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("artifacts/sentiment_pipeline.joblib")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    positive_probability: float
    influential_terms: list[str]


class SentimentModel:
    """A persisted TF-IDF and logistic-regression classifier."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "SentimentModel":
        if len(texts) != len(labels) or len(texts) < 4:
            raise ValueError("training requires matching texts and labels with at least four rows")
        if set(labels) != {"negative", "positive"}:
            raise ValueError("training labels must include positive and negative")

        pipeline = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
                ),
                ("classifier", LogisticRegression(C=10, random_state=0, max_iter=1_000)),
            ]
        )
        pipeline.fit(texts, labels)
        return cls(pipeline)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "SentimentModel":
        return cls(joblib.load(path))

    def save(self, path: str | Path = MODEL_PATH) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, destination)

    def predict(self, text: str) -> Prediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter some review text before analyzing it.")

        probabilities = self._pipeline.predict_proba([text])[0]
        classes = list(self._pipeline.classes_)
        positive_probability = float(probabilities[classes.index("positive")])
        confidence = max(positive_probability, 1 - positive_probability)
        label = "neutral" if confidence < 0.60 else ("positive" if positive_probability >= 0.5 else "negative")
        return Prediction(label, confidence, positive_probability, self._influential_terms(text))

    def _influential_terms(self, text: str) -> list[str]:
        vectorizer: TfidfVectorizer = self._pipeline.named_steps["vectorizer"]
        classifier: LogisticRegression = self._pipeline.named_steps["classifier"]
        features = vectorizer.transform([text])
        contributions = features.multiply(classifier.coef_[0]).toarray()[0]
        terms = vectorizer.get_feature_names_out()
        indexes = sorted(
            (index for index, value in enumerate(contributions) if value != 0),
            key=lambda index: abs(contributions[index]),
            reverse=True,
        )
        return [str(terms[index]) for index in indexes[:3]]
