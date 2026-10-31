"""Train and evaluate the local sentiment model from a labelled CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from model import MODEL_PATH, SentimentModel


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as dataset:
        rows = list(csv.DictReader(dataset))
    if not rows or set(rows[0]) != {"text", "label"}:
        raise ValueError("dataset must have text,label columns")
    return [row["text"] for row in rows], [row["label"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/reviews.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    arguments = parser.parse_args()
    texts, labels = load_dataset(arguments.data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=0, stratify=labels
    )
    model = SentimentModel.train(train_texts, train_labels)
    model.save(arguments.model)
    predictions = [model.predict(text).label for text in test_texts]
    predictions = ["positive" if label == "neutral" else label for label in predictions]
    print(f"Saved model to {arguments.model}")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.2%}")
    print(classification_report(test_labels, predictions, zero_division=0))


if __name__ == "__main__":
    main()
