"""
tests/test_server.py — Unit and integration tests for FastAPI WhatsApp & REST server.

Run with:
    pytest tests/test_server.py -v
"""

import hmac
import hashlib
import json
import sys
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage

from server import (
    app,
    SessionStore,
    MessageDeduplicator,
    verify_meta_signature,
    WHATSAPP_VERIFY_TOKEN,
)
from agent.agent import AgentState, Intent, Lead


@pytest.fixture
def client():
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Health Check Endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthCheck:
    def test_healthz_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "autostream-agent"
        assert "llm_provider" in data


# ══════════════════════════════════════════════════════════════════════════════
# Meta Webhook Handshake (GET /webhook)
# ══════════════════════════════════════════════════════════════════════════════

class TestWebhookVerification:
    def test_valid_handshake_returns_challenge(self, client):
        params = {
            "hub.mode": "subscribe",
            "hub.verify_token": WHATSAPP_VERIFY_TOKEN,
            "hub.challenge": "1158201234",
        }
        resp = client.get("/webhook", params=params)
        assert resp.status_code == 200
        assert resp.text == "1158201234"

    def test_invalid_token_returns_403(self, client):
        params = {
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "1158201234",
        }
        resp = client.get("/webhook", params=params)
        assert resp.status_code == 403

    def test_invalid_mode_returns_403(self, client):
        params = {
            "hub.mode": "unsubscribe",
            "hub.verify_token": WHATSAPP_VERIFY_TOKEN,
            "hub.challenge": "1158201234",
        }
        resp = client.get("/webhook", params=params)
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════════════════════
# Signature Verification & Deduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestSignatureVerification:
    def test_signature_verification_succeeds_with_valid_hash(self, monkeypatch):
        secret = "test_app_secret_123"
        monkeypatch.setattr("server.WHATSAPP_APP_SECRET", secret)

        body = b'{"object":"whatsapp_business_account"}'
        expected_sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        assert verify_meta_signature(body, expected_sig) is True

    def test_signature_verification_fails_with_tampered_body(self, monkeypatch):
        secret = "test_app_secret_123"
        monkeypatch.setattr("server.WHATSAPP_APP_SECRET", secret)

        body = b'{"object":"whatsapp_business_account"}'
        expected_sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        tampered_body = b'{"object":"tampered"}'
        assert verify_meta_signature(tampered_body, expected_sig) is False

    def test_signature_verification_fails_with_malformed_header(self, monkeypatch):
        secret = "test_app_secret_123"
        monkeypatch.setattr("server.WHATSAPP_APP_SECRET", secret)
        assert verify_meta_signature(b"data", "invalid_header") is False


class TestMessageDeduplicator:
    def test_detects_duplicate_message_id(self):
        dedup = MessageDeduplicator()
        assert dedup.is_duplicate("wamid.123") is False
        assert dedup.is_duplicate("wamid.123") is True
        assert dedup.is_duplicate("wamid.456") is False

    def test_empty_message_id_never_duplicate(self):
        dedup = MessageDeduplicator()
        assert dedup.is_duplicate("") is False


# ══════════════════════════════════════════════════════════════════════════════
# Session State Store
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionStore:
    def test_session_isolation(self):
        store = SessionStore()
        state_a = store.get_state("user_a")
        state_b = store.get_state("user_b")

        state_a.lead.name = "Alice"
        store.save_state("user_a", state_a)

        assert store.get_state("user_a").lead.name == "Alice"
        assert store.get_state("user_b").lead.name == ""

    def test_lru_eviction(self):
        store = SessionStore(max_sessions=2)
        store.save_state("s1", AgentState())
        store.save_state("s2", AgentState())
        store.save_state("s3", AgentState())

        assert len(store._sessions) == 2
        assert "s1" not in store._sessions
        assert "s2" in store._sessions
        assert "s3" in store._sessions


# ══════════════════════════════════════════════════════════════════════════════
# REST Chat API (POST /api/chat)
# ══════════════════════════════════════════════════════════════════════════════

class TestRestChatApi:
    def test_empty_message_returns_400(self, client):
        resp = client.post("/api/chat", json={"session_id": "u1", "message": "   "})
        assert resp.status_code == 400

    def test_chat_turn_success(self, client):
        mock_graph = MagicMock()
        mock_output = AgentState(
            intent=Intent.product_inquiry,
            lead_stage="none",
            messages=[
                HumanMessage(content="How much is Pro?"),
                AIMessage(content="Pro plan is $79/month."),
            ]
        )
        mock_graph.ainvoke = AsyncMock(return_value=mock_output)

        with patch("server._GRAPH", mock_graph):
            resp = client.post("/api/chat", json={"session_id": "test_user", "message": "How much is Pro?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Pro plan is $79/month."
        assert data["intent"] == "product_inquiry"
        assert data["lead_stage"] == "none"
