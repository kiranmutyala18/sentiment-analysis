"""FastAPI service and HTML interface for the sentiment models."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from model import MODEL_PATH, SentimentModel
from storage import PredictionStore


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


def create_app(model_path: str | Path = MODEL_PATH, database_path: str | Path = "data/predictions.db") -> FastAPI:
    model = SentimentModel.load(model_path)
    store = PredictionStore(database_path)
    templates = Jinja2Templates(directory="templates")
    app = FastAPI(title="Review Sentiment Service", version="0.3.0")

    def analyze_text(text: str) -> dict[str, object]:
        prediction = model.predict(text)
        store.save(text, prediction)
        return {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "positive_probability": prediction.positive_probability,
            "influential_terms": prediction.influential_terms,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"history": store.recent()})

    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze_form(request: Request, text: str = Form()) -> HTMLResponse:
        try:
            result = analyze_text(text)
        except ValueError as error:
            return templates.TemplateResponse(request, "index.html", {"text": text, "error": str(error), "history": store.recent()}, status_code=422)
        return templates.TemplateResponse(request, "index.html", {"text": text, "prediction": result, "history": store.recent()})

    @app.post("/api/v1/analyze")
    async def analyze_api(request: AnalyzeRequest) -> dict[str, object]:
        try:
            return analyze_text(request.text)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/history")
    async def history() -> list[dict[str, object]]:
        return store.recent()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
