import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Structural Drawing AI System"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_inference_image_validation():
    # Test that uploading a non-image file fails gracefully
    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("test.txt", b"dummy text", "text/plain")}
    )
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_inference_pdf_validation():
    # Test that uploading a non-pdf file to the pdf endpoint fails gracefully
    response = client.post(
        "/api/v1/inference/pdf",
        files={"file": ("test.png", b"dummy image", "image/png")}
    )
    assert response.status_code == 400
    assert "must be a PDF" in response.json()["detail"]

def test_dashboard_summary():
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload
    assert "splits" in payload
    assert "sources" in payload

def test_sample_index_and_detail():
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    samples = response.json()["samples"]
    if not samples:
        pytest.skip("No sample files are available in this checkout")

    detail = client.get(f"/api/v1/samples/{samples[0]['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert "labels" in detail_payload
    assert "semantics" in detail_payload

def test_sample_image_endpoint():
    response = client.get("/api/v1/samples")
    samples = response.json()["samples"]
    if not samples:
        pytest.skip("No sample files are available in this checkout")

    image = client.get(f"/api/v1/samples/{samples[0]['id']}/image?boxes=true")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
