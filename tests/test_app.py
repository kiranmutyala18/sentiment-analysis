from model import SentimentModel
from app import create_app


def test_json_api_returns_prediction(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    SentimentModel.train(
        ["excellent quality", "great product", "terrible quality", "broken product"],
        ["positive", "positive", "negative", "negative"],
    ).save(model_path)
    client = create_app(model_path, tmp_path / "predictions.db").test_client()

    response = client.post("/api/v1/analyze", json={"text": "excellent product"})

    assert response.status_code == 200
    assert response.json["label"] == "positive"
    assert response.json["influential_terms"]
