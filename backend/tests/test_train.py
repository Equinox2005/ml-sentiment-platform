import json

import pytest
from fastapi.testclient import TestClient

import main
import train


@pytest.mark.parametrize(
    "text",
    [
        "I LOVE this!!! http://x.co @bob #blessed",
        "Don't   stop\nme now",
        "123 :) ok",
        "",
    ],
)
def test_serving_cleans_text_like_training(text):
    assert main.clean_text(text) == train.clean_text(text)


def test_demo_model_trains_and_can_be_served(tmp_path, monkeypatch):
    artifacts = train.train_model(train.load_demo_data())
    train.save_artifacts(artifacts, output_dir=str(tmp_path))

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["label_map"] == {"0": "negative", "1": "positive", "2": "neutral"}

    monkeypatch.setattr(main, "MODEL_DIR", str(tmp_path))
    with TestClient(main.app) as client:
        body = client.post("/predict", json={"text": "terrible, a complete waste of money"}).json()

    assert set(body["probabilities"]) == {"negative", "positive", "neutral"}
    words = {kw["word"]: kw["type"] for kw in body["key_words"]}
    assert words["terrible"] == "negative"
