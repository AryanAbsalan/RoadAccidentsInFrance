import pytest
from fastapi.testclient import TestClient

import sys
import os

# Make sure the src directory is in the sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.app.main import app  

@pytest.fixture(scope="module")
def client():
    # Setup the TestClient for the FastAPI app
    client = TestClient(app)
    yield client
    
def test_homepage(client):
    response = client.get("/")  # Assuming your homepage is mapped to "/"
    assert response.status_code == 200  # Assert that the status code is 200 OK
    assert "Welcome to the API" in response.text  

# Test case to check if the FastAPI app is created successfully
def test_app_creation():
    # Send a GET request to the root endpoint
    response = client.get("/")
    
    # Assert that the status code is 200, indicating that the app is running
    assert response.status_code == 200
    assert "Welcome To The Road Accidents In France API" in response.text  


