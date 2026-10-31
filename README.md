# Review Sentiment Service

A trainable, explainable sentiment-analysis service for product reviews. It provides a Flask interface, JSON API, SQLite-backed prediction history, persisted model artifact, and a reproducible training workflow.

## Architecture

```text
reviews.csv -> train.py -> persisted scikit-learn pipeline
                                  |
web UI / JSON API -> Flask -> prediction + explanation -> SQLite history
```

## Features

- Reproducible training from a labelled `text,label` CSV
- TF-IDF feature extraction and logistic regression classifier
- Persisted model artifact using `joblib`
- Positive, negative, and uncertainty-based neutral output
- Confidence and influential-term explanation for every prediction
- JSON API and saved recent predictions
- Automated tests for model persistence and API behavior

## Run it

```powershell
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open `http://127.0.0.1:5000`.

## API

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/v1/analyze `
  -ContentType 'application/json' `
  -Body '{"text":"Excellent support and reliable quality"}'
```

The response contains `label`, `confidence`, `positive_probability`, and `influential_terms`.

## Model choices

TF-IDF turns text into weighted word and phrase features. Logistic regression is fast, produces probabilities, and lets the app show the terms that affected a prediction. A probability close to 50% is presented as neutral to avoid overclaiming certainty.

## Limitations

The starter dataset is deliberately small and transparent. A production system needs a representative labelled dataset, evaluation across relevant user groups, monitoring, and protections against input abuse. It can also miss sarcasm and domain-specific language.
