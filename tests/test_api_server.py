"""Smoke test for the OpenAI-compatible PRISM API server.

The server exposes /v1/chat/completions with text + image_url parts only.
This test exercises the lightweight surface (the /v1/models route and the
request validation path) without requiring a real PRISM checkpoint.

Loading the model would need PRISM_CHECKPOINT to point at a real
checkpoint directory; that path is covered by an integration test
elsewhere (gated behind --runslow).
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import server as server_mod  # noqa: E402

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(monkeypatch):
    # Skip the lifespan model load — we only exercise routes that don't
    # touch the model state, and those that do are validated to fail loudly.
    monkeypatch.setattr(server_mod, "load_model", lambda: None)
    monkeypatch.setattr(server_mod, "model", None)
    with TestClient(server_mod.app) as c:
        yield c


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert any(m["id"] == server_mod.SERVED_MODEL_ID for m in data["data"])


def test_chat_completions_rejects_empty_messages(client):
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert "messages must not be empty" in response.text
