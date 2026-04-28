from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_route_requires_api_key():
    response = client.get("/monitoring/summary")
    assert response.status_code == 401
