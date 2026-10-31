"""Flask application for local product-review sentiment analysis."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model import MODEL_PATH, SentimentModel
from storage import PredictionStore


def create_app(model_path: str | Path = MODEL_PATH, database_path: str | Path = "data/predictions.db") -> Flask:
    app = Flask(__name__)
    model = SentimentModel.load(model_path)
    store = PredictionStore(database_path)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", history=store.recent())

    @app.post("/analyze")
    def analyze() -> str:
        text = request.form.get("text", "")
        try:
            prediction = model.predict(text)
        except ValueError as error:
            return render_template("index.html", text=text, error=str(error), history=store.recent()), 400
        store.save(text, prediction)
        return render_template("index.html", text=text, prediction=prediction, history=store.recent())

    @app.post("/api/v1/analyze")
    def api_analyze():
        body = request.get_json(silent=True) or {}
        try:
            prediction = model.predict(body.get("text", ""))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        store.save(body["text"], prediction)
        return jsonify(
            label=prediction.label,
            confidence=round(prediction.confidence, 4),
            positive_probability=round(prediction.positive_probability, 4),
            influential_terms=prediction.influential_terms,
        )

    @app.get("/api/v1/history")
    def api_history():
        return jsonify(store.recent())

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
