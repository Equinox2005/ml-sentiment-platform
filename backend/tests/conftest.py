import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND_DIR), str(BACKEND_DIR.parent / "model")]

import main  # noqa: E402


@pytest.fixture
def client():
    """API client serving the model committed in model/."""
    main.STATS = main.Stats()
    with TestClient(main.app) as test_client:
        yield test_client
