import pytest
from fastapi.testclient import TestClient
from app.main import UserModel, PredictionInput
from unittest.mock import MagicMock

from app.main import get_current_active_user, get_db  
from app.main import get_db, get_password_hash  # Adjust as needed for imports
from src.app.model_handler import ModelHandler as model_handler

import sys
import os

# Make sure the src directory is in the sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app.main import app  

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
def test_app_creation(client):
    # Send a GET request to the root endpoint
    response = client.get("/")
    
    # Assert that the status code is 200, indicating that the app is running
    assert response.status_code == 200
    assert "Welcome To The Road Accidents In France API" in response.text  

# Mocking the database functions
@pytest.fixture(scope="module")
def mock_db():
    mock_db = MagicMock()  # Creating a mock for DB session
    yield mock_db


def test_register_Username_already_registered(client, mock_db):
    # Simulate that the user already exists in the database by mocking the database query
    # This mock ensures that when filter_by() is called, it returns a mock with the `first()` method returning a user object
    mock_user = {"username":'new_user', "id":1, "disabled":False}
    mock_db.query.return_value.filter_by.return_value.first(mock_user) #= mock_user

    # Input user data (the same username that already exists in the database)
    user_data = {
        "username": "new_user",
        "hashed_password": "securepassword123",
        "disabled": False
    }

    # Mock the get_db dependency to return our mock_db
    app.dependency_overrides[get_db] = lambda: mock_db

    # Make a POST request to register the new user with the same username that already exists
    response = client.post("/register/", json=user_data)

    # Assert that the response status code is 400 (Bad Request)
    assert response.status_code == 400, f"Expected 400 but got {response.status_code}. Response body: {response.json()}"

    # Assert that the response body contains the 'detail' key with the expected error message
    assert response.json() == {"detail": "Username already registered"}

    # Verify that the database query was made to check if the user exists
    mock_db.query.return_value.filter_by.return_value.first.assert_called_once_with({'username': 'new_user', 'id': 1, 'disabled': False})

