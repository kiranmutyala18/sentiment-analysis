from model import SentimentModel


def model() -> SentimentModel:
    return SentimentModel.train(
        ["excellent and reliable", "great quality", "terrible and broken", "poor quality"],
        ["positive", "positive", "negative", "negative"],
    )


def test_positive_text_is_positive() -> None:
    assert model().predict("excellent quality").label == "positive"


def test_negative_text_is_negative() -> None:
    assert model().predict("terrible broken product").label == "negative"


def test_model_can_be_saved_and_loaded(tmp_path) -> None:
    location = tmp_path / "model.joblib"
    model().save(location)

    assert SentimentModel.load(location).predict("great quality").label == "positive"
