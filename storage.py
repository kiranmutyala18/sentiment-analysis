"""SQLite-backed prediction history."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from model import Prediction


class PredictionStore:
    def __init__(self, path: str | Path = "data/predictions.db") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY, text TEXT NOT NULL, label TEXT NOT NULL,
                    confidence REAL NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def save(self, text: str, prediction: Prediction) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO predictions (text, label, confidence) VALUES (?, ?, ?)",
                (text, prediction.label, prediction.confidence),
            )

    def recent(self, limit: int = 8) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT text, label, confidence, created_at FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection
