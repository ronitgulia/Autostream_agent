"""
server.py — Production FastAPI webhook server for AutoStream AI Agent.

Provides:
  1. WhatsApp Cloud API Webhook Integration:
     - GET /webhook: Verification challenge handshake for Meta Developer Portal.
     - POST /webhook: Webhook event receiver with HMAC SHA-256 signature verification,
       message deduplication, per-user state persistence, and background agent execution.
  2. REST API:
     - POST /api/chat: Direct JSON chat endpoint for web/mobile applications.
     - GET /healthz: Health check and runtime telemetry.

Run with:
    uvicorn server.py:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import hmac
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.agent import build_graph, AgentState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("autostream.server")


# ── Configuration ──────────────────────────────────────────────────────────────

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "autostream_verify_token")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v19.0")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_SESSION_TTL_SECONDS = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "86400"))


# ── State Serialization Helpers ────────────────────────────────────────────────

def state_to_dict(state: AgentState) -> dict:
    """Serialize an AgentState object into a JSON-compatible dictionary."""
    data = state.model_dump(exclude={"messages"})
    messages_data = []
    for msg in state.messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai" if isinstance(msg, AIMessage) else "system"
        messages_data.append({"role": role, "content": msg.content})
    data["messages"] = messages_data
    return data


def state_from_dict(data: dict) -> AgentState:
    """Deserialize a dictionary back into an AgentState instance."""
    messages = []
    for m in data.get("messages", []):
        role = m.get("role", "human")
        content = m.get("content", "")
        if role == "ai":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    state_data = {k: v for k, v in data.items() if k != "messages"}
    return AgentState(messages=messages, **state_data)


# ── Session State Persistence & Locks ──────────────────────────────────────────

class SessionStore:
    """Thread-safe session store with in-memory LRU and optional Redis persistence."""

    def __init__(self, max_sessions: int = 10000, redis_url: Optional[str] = None) -> None:
        self._sessions: OrderedDict[str, AgentState] = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._max_sessions = max_sessions
        self._redis_url = redis_url or REDIS_URL
        self._redis_client = None

        if self._redis_url:
            try:
                import redis
                self._redis_client = redis.Redis.from_url(self._redis_url, decode_responses=True)
                self._redis_client.ping()
                logger.info("Connected to distributed Redis session store at %s", self._redis_url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); falling back to in-memory store.", exc)
                self._redis_client = None

    def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def get_state(self, session_id: str) -> AgentState:
        if self._redis_client:
            try:
                raw = self._redis_client.get(f"autostream:session:{session_id}")
                if raw:
                    return state_from_dict(json.loads(raw))
            except Exception as exc:
                logger.warning("Failed to fetch state from Redis for %s: %s", session_id, exc)

        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        new_state = AgentState()
        self.save_state(session_id, new_state)
        return new_state

    def save_state(self, session_id: str, state: AgentState) -> None:
        if self._redis_client:
            try:
                payload = json.dumps(state_to_dict(state))
                self._redis_client.setex(
                    f"autostream:session:{session_id}",
                    REDIS_SESSION_TTL_SECONDS,
                    payload,
                )
            except Exception as exc:
                logger.warning("Failed to persist state to Redis for %s: %s", session_id, exc)

        self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        if len(self._sessions) > self._max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            self._locks.pop(evicted_id, None)


# ── Message Deduplication ──────────────────────────────────────────────────────

class MessageDeduplicator:
    """Tracks recently processed WhatsApp message IDs (wamid) to ignore retries."""

    def __init__(self, max_items: int = 5000) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max_items = max_items

    def is_duplicate(self, message_id: str) -> bool:
        if not message_id:
            return False
        if message_id in self._seen:
            return True
        self._seen[message_id] = time.time()
        if len(self._seen) > self._max_items:
            self._seen.popitem(last=False)
        return False


# ── Global State ───────────────────────────────────────────────────────────────

_SESSION_STORE = SessionStore()
_DEDUPLICATOR = MessageDeduplicator()
_GRAPH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _GRAPH
    logger.info("Initializing AutoStream LangGraph state machine...")
    _GRAPH = build_graph()
    logger.info("AutoStream agent ready to serve requests.")
    yield
    logger.info("Shutting down AutoStream webhook server.")


app = FastAPI(
    title="AutoStream AI Agent Server",
    description="WhatsApp Cloud API Webhook and REST API for AutoStream sales assistant",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Signature Verification ─────────────────────────────────────────────────────

def verify_meta_signature(payload: bytes, signature_header: Optional[str]) -> bool:
    """Verify Meta X-Hub-Signature-256 header using SHA256 HMAC."""
    if not WHATSAPP_APP_SECRET:
        # If secret is not configured, skip verification with a warning
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_hash = signature_header[7:].strip()
    computed_hash = hmac.new(
        key=WHATSAPP_APP_SECRET.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed_hash, expected_hash)


# ── WhatsApp Outbound Client ───────────────────────────────────────────────────

async def send_whatsapp_message(to_phone: str, message_text: str) -> bool:
    """Send text message to a user via Meta's WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning(
            "WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set. Mocking reply to %s: %s",
            to_phone, message_text
        )
        return False

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone,
        "type": "text",
        "text": {"preview_url": False, "body": message_text},
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                logger.info("Successfully sent WhatsApp message to %s", to_phone)
                return True
            else:
                logger.error(
                    "Failed to send WhatsApp message to %s: HTTP %d - %s",
                    to_phone, resp.status_code, resp.text
                )
                return False
    except Exception as exc:
        logger.exception("Exception while calling WhatsApp Cloud API: %s", exc)
        return False


# ── Background Message Processor ──────────────────────────────────────────────

async def process_whatsapp_turn(phone_number: str, user_text: str) -> None:
    """Process user turn through the LangGraph agent and dispatch reply."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    lock = _SESSION_STORE.get_lock(phone_number)
    async with lock:
        current_state = _SESSION_STORE.get_state(phone_number)

        updated_state = current_state.model_copy(
            update={"messages": current_state.messages + [HumanMessage(content=user_text)]}
        )

        try:
            result = await _GRAPH.ainvoke(updated_state)
            if isinstance(result, dict):
                final_state = AgentState(**result)
            elif isinstance(result, AgentState):
                final_state = result
            else:
                final_state = updated_state

            _SESSION_STORE.save_state(phone_number, final_state)

            last_ai_msg = next(
                (m.content for m in reversed(final_state.messages) if isinstance(m, AIMessage)),
                "Thanks for reaching out! How can I help you today?"
            )

            await send_whatsapp_message(phone_number, last_ai_msg)

        except Exception as exc:
            logger.exception("Error processing turn for %s: %s", phone_number, exc)
            fallback = "I had a momentary connection hiccup. Please send your message again! 🙏"
            await send_whatsapp_message(phone_number, fallback)


# ── REST API Schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Unique user or conversation identifier")
    message: str = Field(..., description="User message content")


class ChatResponse(BaseModel):
    reply: str
    intent: str
    lead_stage: str
    lead_collected: dict


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def health_check():
    """Health status and configuration telemetry."""
    return {
        "status": "healthy",
        "service": "autostream-agent",
        "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
        "llm_model": os.getenv("LLM_MODEL", "provider-default"),
        "whatsapp_configured": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "redis_connected": bool(_SESSION_STORE._redis_client is not None),
    }


@app.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """Handshake verification for Meta Developer Portal."""
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verification challenge successful.")
        return Response(content=hub_challenge or "", media_type="text/plain")

    logger.warning("Webhook verification failed: token mismatch or invalid mode.")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Incoming event notification receiver from Meta WhatsApp Cloud API."""
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_meta_signature(body, signature):
        logger.warning("Rejected webhook request: invalid signature.")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Meta WhatsApp Cloud API payload traversal
    entries = data.get("entry", [])
    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                msg_id = msg.get("id")
                sender = msg.get("from")
                msg_type = msg.get("type")

                if _DEDUPLICATOR.is_duplicate(msg_id):
                    logger.info("Ignoring duplicate message ID: %s", msg_id)
                    continue

                if msg_type == "text" and sender:
                    text_body = msg.get("text", {}).get("body", "").strip()
                    if text_body:
                        logger.info("Received message from %s: '%s'", sender, text_body)
                        background_tasks.add_task(process_whatsapp_turn, sender, text_body)

    return {"status": "accepted"}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(payload: ChatRequest):
    """Direct REST chat API for web/mobile applications."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    session_id = payload.session_id.strip() or "default"
    user_text = payload.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    lock = _SESSION_STORE.get_lock(session_id)
    async with lock:
        state = _SESSION_STORE.get_state(session_id)
        state = state.model_copy(
            update={"messages": state.messages + [HumanMessage(content=user_text)]}
        )

        result = await _GRAPH.ainvoke(state)
        if isinstance(result, dict):
            final_state = AgentState(**result)
        elif isinstance(result, AgentState):
            final_state = result
        else:
            final_state = state

        _SESSION_STORE.save_state(session_id, final_state)

        last_ai_msg = next(
            (m.content for m in reversed(final_state.messages) if isinstance(m, AIMessage)),
            "Thank you for contacting AutoStream!"
        )

        return ChatResponse(
            reply=last_ai_msg,
            intent=str(final_state.intent.value if hasattr(final_state.intent, 'value') else final_state.intent),
            lead_stage=final_state.lead_stage,
            lead_collected=final_state.lead.model_dump(),
        )
