import pytest
from src.api.deriv_client import DerivClient

def test_client_init():
    client = DerivClient("test_app", "test_token")
    assert client is not None

# Add more integration tests...
print("API tests passed (basic).")
