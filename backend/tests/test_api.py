import pytest


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_model_info_describes_shipped_model(client):
    body = client.get("/model").json()
    assert body["labels"] == ["negative", "positive"]
    assert body["vocabulary_size"] == 50000
    assert body["dataset"] == "Sentiment140"
    assert body["accuracy"] == pytest.approx(0.818, abs=0.001)


def test_predict_positive(client):
    body = client.post("/predict", json={"text": "I love this, it's amazing"}).json()
    assert body["prediction"] == "positive"
    assert body["confidence"] > 0.9
    assert set(body["probabilities"]) == {"negative", "positive"}
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)


def test_predict_negative(client):
    body = client.post("/predict", json={"text": "This is the worst thing ever"}).json()
    assert body["prediction"] == "negative"


def test_key_words_explain_the_prediction(client):
    body = client.post("/predict", json={"text": "not good at all"}).json()
    assert body["prediction"] == "negative"
    assert body["key_words"][0]["word"] == "not good"
    assert body["key_words"][0]["type"] == "negative"

    words = {kw["word"]: kw["type"] for kw in body["key_words"]}
    assert words["good"] == "positive"


@pytest.mark.parametrize("text", ["   ", "!!! 123", "@someone #tag https://example.com"])
def test_predict_rejects_text_with_nothing_to_analyze(client, text):
    res = client.post("/predict", json={"text": text})
    assert res.status_code == 400
    assert "Nothing left to analyze" in res.json()["detail"]


@pytest.mark.parametrize("text", ["", "a" * 5001])
def test_predict_validates_length(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422


def test_batch_skips_texts_with_nothing_to_analyze(client):
    res = client.post("/predict/batch", json={"texts": ["great day", "123 :)", "awful day"]})
    assert res.status_code == 200
    body = res.json()
    assert [r["text"] for r in body["results"]] == ["great day", "awful day"]
    assert [r["prediction"] for r in body["results"]] == ["positive", "negative"]
    assert body["skipped"] == [1]


def test_batch_size_limit(client):
    assert client.post("/predict/batch", json={"texts": ["ok"] * 51}).status_code == 422
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_compare_returns_probability_delta(client):
    body = client.post(
        "/compare", json={"text_a": "best day ever", "text_b": "worst day ever"}
    ).json()
    a, b = body["result_a"], body["result_b"]
    assert (a["prediction"], b["prediction"]) == ("positive", "negative")
    assert body["delta"]["positive"] == pytest.approx(
        a["probabilities"]["positive"] - b["probabilities"]["positive"], abs=1e-3
    )


def test_compare_says_which_text_is_unusable(client):
    res = client.post("/compare", json={"text_a": "fine", "text_b": "12345"})
    assert res.status_code == 400
    assert res.json()["detail"].startswith("text_b:")


def test_stats_keep_counts_but_not_text(client):
    client.post("/predict", json={"text": "I love it"})
    client.post("/predict/batch", json={"texts": ["I hate it", "!!!"]})
    client.post("/predict", json={"text": "???"})

    body = client.get("/stats").json()
    assert body["total_predictions"] == 2
    assert body["label_distribution"] == {"positive": 1, "negative": 1}
    assert "love" not in str(body)
