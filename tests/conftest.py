import os
import tempfile
from pathlib import Path

import pytest

_TEMP_DIR = Path(tempfile.mkdtemp(prefix="diabetes-test-"))
os.environ["DATA_DIR"] = str(_TEMP_DIR)
os.environ["DATABASE_URL"] = "sqlite:///" + (_TEMP_DIR / "predictions.db").as_posix()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
