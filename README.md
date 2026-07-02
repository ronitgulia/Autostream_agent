 # AutoStream AI Agent 

> **Social-to-Lead Agentic Workflow** | ServiceHive × Inflx Assignment  
> A production-grade conversational AI agent that handles product queries, detects high-intent users, and captures leads automatically.

---

##  Project Structure

```
autostream-agent/
├── main.py                          # CLI entrypoint — token streaming terminal interface
├── server.py                        # FastAPI server — WhatsApp Cloud API webhook & REST API
├── requirements.txt
├── Dockerfile                       # Multi-stage hardened production Docker container
├── docker-compose.yml               # Service orchestration (agent + Redis 7)
├── .env.example                     # Environment configuration template
│
├── agent/
│   └── agent.py                     # LangGraph graph definition (nodes + state)
│
├── tools/
│   └── tools.py                     # Hybrid RAG (Chroma + BM25) + mock lead capture tool
│
├── eval/
│   ├── scenarios.json               # Multi-turn persona dialogue test dataset
│   └── benchmark.py                 # Automated evaluation harness & latency metrics
│
├── tests/
│   ├── test_agent.py                # Unit tests for state machine, RAG, and lead capture
│   ├── test_server.py               # Integration tests for FastAPI webhook & Redis session store
│   └── test_benchmark.py            # Unit tests for benchmark runner & persona simulators
│
├── .github/workflows/
│   └── ci.yml                       # Multi-python CI matrix (3.10 - 3.13) + flake8 linting
│
└── knowledge_base/
    └── autostream_kb.json           # Local knowledge base (pricing, policies, FAQs)
```

---

##  How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your preferred LLM provider and API key:

```env
LLM_PROVIDER=anthropic            # Options: anthropic | openai | google | groq
# Optional model override (defaults: claude-3-5-haiku-latest, gpt-4o-mini, gemini-2.0-flash, llama-3.3-70b-versatile)
# LLM_MODEL=claude-3-5-haiku-latest

ANTHROPIC_API_KEY=sk-ant-...      # Claude 3.5 Haiku (default)
# OPENAI_API_KEY=sk-...           # GPT-4o-mini (alternative)
# GOOGLE_API_KEY=AIza...          # Gemini 2.0 Flash (alternative)
# GROQ_API_KEY=gsk_...            # Llama-3.3-70b (alternative)
```

### 5. Run the agent

```bash
# Run with defaults
python main.py

# Or customize provider, model, and logging via CLI flags
python main.py --provider openai --model gpt-4o-mini
python main.py --debug
```

You'll see an interactive CLI with real-time token streaming. Type your messages and press Enter. Type `quit` to exit.

---

##  Example Conversation Flow

```
You: Hi there!
Agent: Hey! Welcome to AutoStream  — the AI-powered video editor for creators...

You: What are your pricing plans?
Agent: Great question! Here's what we offer:
   Basic Plan — $29/month: 10 videos, 720p export...
   Pro Plan — $79/month: Unlimited videos, 4K, AI captions...

You: That sounds great, I want to try the Pro plan for my YouTube channel.
Agent: Great to hear you're interested! Could I start with your full name?

You: Riya Sharma
Agent: Nice to meet you, Riya! What's your email address?

You: riya@gmail.com
Agent: Got it! Which platform do you primarily create content on?

You: YouTube
Agent:  Lead captured! Our team will reach out to riya@gmail.com within 24 hours...
```

---

##  Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it offers **explicit, inspectable state machines** ideal for multi-step agentic workflows. Unlike AutoGen's agent-to-agent paradigm (optimised for collaborative multi-agent tasks), LangGraph lets us define a deterministic graph of nodes with conditional routing — perfect for a workflow where we must carefully control *when* tool calls happen (e.g., not triggering lead capture prematurely).

### Graph Structure

The agent is built as a **directed state graph** with four nodes:

1. **`intent_detector`** — Every user message first passes through a zero-temperature LLM intent classifier with few-shot anchoring that deterministically labels it as `greeting`, `product_inquiry`, or `high_intent`. Built-in circuit breaking trips if consecutive LLM failures exceed threshold.
2. **`greeter`** — Handles casual conversation using a system-prompted LLM response.
3. **`rag_answer`** — Runs parallel retrieval and history trimming, injecting hybrid RAG context into the LLM prompt. Also serves post-signup questions with reassuring context.
4. **`lead_capture`** — Structured form extractor validating fields with Pydantic (`EmailStr`). Supports **interruptible conversation flow**: if a user asks a product question mid-collection, the agent retrieves RAG context, answers accurately, and gently resumes collecting missing fields. Only triggers `mock_lead_capture()` once all three fields are valid.

### State & Memory Management

- **Pydantic Validation**: Full `AgentState` model backed by Pydantic v2 ensuring strict runtime validation across all nodes.
- **Sliding-Window Memory**: Context trimming automatically summarizes exchanges older than `KEEP_LAST_N=10` turns into a system summary while pruning historical messages, preventing quadratic token inflation over extended dialogues.
- **Hybrid RAG & Lazy Loading**: Combines dense semantic search (ChromaDB + `all-MiniLM-L6-v2`) and sparse keyword matching (BM25) fused via Reciprocal Rank Fusion (RRF). Lazy thread-safe singletons ensure near-instant cold start without blocking module imports.

---

##  WhatsApp Deployment via Webhooks & REST Server

The repository includes a production-ready **FastAPI webhook server** (`server.py`) that implements the full Meta WhatsApp Cloud API flow and a standalone REST chat API.

### Running the Server

```bash
# Start the webhook & REST server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

* `GET /healthz`: Health status and active LLM configuration.
* `GET /webhook`: Webhook verification challenge handshake for the Meta Developer Portal (`hub.challenge`).
* `POST /webhook`: Webhook event receiver with `X-Hub-Signature-256` HMAC validation, message deduplication, and async LangGraph execution.
* `POST /api/chat`: REST endpoint for web or mobile clients:
  ```bash
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"session_id": "user_123", "message": "What are your pricing plans?"}'
  ```

### Architecture Flow

```
WhatsApp User
     │  (sends message)
     ▼
WhatsApp Cloud API (Meta)
     │  HTTP POST (with X-Hub-Signature-256)
     ▼
FastAPI Server (server.py)
  ├── Verifies HMAC SHA-256 signature against WHATSAPP_APP_SECRET
  ├── Deduplicates incoming message ID (wamid)
  ├── Acknowledges Meta immediately with 200 OK
  ├── Dispatches background task:
  │     ├── Retrieves/initializes user AgentState from SessionStore
  │     ├── Invokes LangGraph agent state machine
  │     ├── Persists updated AgentState for the phone number
  │     └── Sends AI response back via WhatsApp Cloud API POST /messages
     │
     ▼
WhatsApp User receives reply
```

---

## 🐳 Docker & Distributed Session Storage

AutoStream Agent is containerized with multi-stage build caching, non-root security hardening, and distributed Redis state persistence.

### Run with Docker Compose

Spin up both the FastAPI Agent server and a dedicated Redis 7 session store with a single command:

```bash
docker compose up -d --build
```

- **Agent Server**: Accessible on port `8000` with container healthchecks at `http://localhost:8000/healthz`.
- **Redis Session Store**: Accessible on port `6379`.
- **Automatic Fallback**: If Redis is not configured or temporarily unreachable, `SessionStore` automatically fails over to an in-memory LRU cache with zero downtime.

---

## 📊 Evaluation & Persona Benchmarking

To ensure reliability, intent accuracy, and lead conversion resilience across multi-turn interactions, the repository provides an automated simulation benchmark suite.

### Running the Benchmark

```bash
# Fast offline simulation (ideal for CI, no API keys required):
python eval/benchmark.py --mock-llm

# Live evaluation against configured LLMs with Markdown export:
python eval/benchmark.py --scenarios eval/scenarios.json --output benchmark_report.md
```

### Simulated Persona Scenarios (`eval/scenarios.json`)

1. **Curious Creator**: Multi-turn pricing and feature inquiry followed by seamless lead capture.
2. **Skeptical Evaluator (Interruption Recovery)**: User pauses lead capture mid-flow to ask about refund policies. The agent answers via hybrid RAG and gracefully resumes collection.
3. **Direct Conversion**: One-shot lead submission with complete contact details.
4. **Casual Greeter**: Low-intent exploration and product overview.

Metrics measured:
- **Scenario Pass Rate (%)**: Percentage of multi-turn scenarios meeting expected final intent and lead capture criteria.
- **Intent Accuracy (%)**: Turn-by-turn intent classification precision.
- **Latency (ms)**: Turn latency, average duration, and P95 latency.

---

## 🧪 Testing & CI/CD

Run the comprehensive test suite across unit, server integration, and evaluation components:

```bash
# Run all unit and integration tests
pytest -v

# Run specific test suites
pytest tests/test_agent.py -v       # Graph routing, RAG, and memory tests
pytest tests/test_server.py -v      # Webhook, HMAC auth, and Redis store tests
pytest tests/test_benchmark.py -v   # Benchmark runner and simulation tests
```

Continuous integration is enforced via **GitHub Actions** across a multi-python matrix (`3.10`, `3.11`, `3.12`, `3.13`) alongside flake8 linting on every push and pull request.

---

##  Evaluation Criteria Addressed

| Criterion | Implementation |
|---|---|
| Intent detection | LLM classifier node with temperature=0 and few-shot examples |
| RAG | Hybrid retrieval (ChromaDB dense + BM25 sparse fused with RRF) with lazy loading |
| State management | `AgentState` Pydantic v2 model with sliding-window summarization |
| Tool calling logic | `mock_lead_capture()` fires once all 3 fields are valid; persists to `leads.csv` |
| Code clarity | Clean architecture: `agent/`, `tools/`, `knowledge_base/`, typed schemas |
| Conversational flow | Handles interruptions & product inquiries seamlessly during lead collection |
| Deployability | WhatsApp webhook architecture documented above |

---

##  Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Type hinting, async generators, modern standard library |
| Framework | LangGraph + LangChain Core | Deterministic state machine, graph routing, token streaming |
| LLM | Claude 3.5 Haiku (default) / OpenAI / Gemini / Groq | Pluggable multi-provider factory with `LLM_MODEL` overrides |
| Vector Store | ChromaDB + sentence-transformers | Local persistent vector search (`all-MiniLM-L6-v2`) |
| Sparse Search | Rank-BM25 | Keyword lexical retrieval fused via Reciprocal Rank Fusion |
| Data Validation | Pydantic v2 + EmailStr | Robust schema validation with automatic email normalization |

---

##  License

MIT — built as a technical assignment for ServiceHive / Inflx.
