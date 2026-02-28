from fastapi.testclient import TestClient

from app.api import app


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_api_feedback() -> None:
    client = TestClient(app)
    response = client.post(
        "/feedback",
        json={
            "company": "BIM",
            "quarter": "Q1",
            "metric": "net_kar",
            "extracted_value": "1,23 mlr TL",
            "user_value": "1,20 mlr TL",
            "evidence_ref": "[doc|Q1|5|gelir tablosu]",
            "verdict": "yanlis",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "feedback_saved"
