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
