# Engineering notes

- `train.py` is separated from the web application so retraining is intentional and repeatable.
- The saved pipeline includes the vectorizer and classifier. Production predictions therefore use the exact same feature transformation as training.
- SQLite records each prediction so the web interface can show recent activity without another service.
- The API accepts JSON at `POST /api/v1/analyze`; `GET /api/v1/history` exposes recent saved records.
- Neutral is an uncertainty threshold, not a third class in the training data.
