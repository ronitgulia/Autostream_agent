 # AutoStream AI Agent 

> **Social-to-Lead Agentic Workflow** | ServiceHive × Inflx Assignment  
> A production-grade conversational AI agent that handles product queries, detects high-intent users, and captures leads automatically.

---

##  Project Structure

```
autostream-agent/
├── main.py                          # CLI entrypoint — run this to start the agent
├── requirements.txt
├── .env.example                     # Copy to .env and add your API key
│
├── agent/
│   └── agent.py                     # LangGraph graph definition (nodes + state)
│
├── tools/
│   └── tools.py                     # RAG retrieval + mock lead capture tool
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

##  WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following architecture would be used:

### Flow

```
WhatsApp User
     │  (sends message)
     ▼
WhatsApp Cloud API (Meta)
     │  HTTP POST → Webhook
     ▼
Your Backend Server (FastAPI / Flask)
  ├── Verifies webhook signature (X-Hub-Signature-256)
  ├── Extracts message body + sender phone number
  ├── Loads user's AgentState from a persistent store (Redis / DynamoDB)
  ├── Calls graph.invoke(state) with the new HumanMessage
  ├── Saves updated state back to the store (keyed by phone number)
  └── Sends AI response back via WhatsApp Cloud API (POST /messages)
     │
     ▼
WhatsApp User receives reply
```

### Key Implementation Steps

1. **Register a Webhook** at `https://your-server.com/webhook` in the Meta Developer Portal. Meta will send a `GET` request with a `hub.challenge` token that your server must echo back to verify ownership.

2. **Handle Incoming Messages**: On each `POST` from Meta, parse the payload:
   ```python
   body = request.json()
   phone = body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
   text  = body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
   ```

3. **Persist State per User**: Use Redis with `phone` as the key to store and retrieve each user's `AgentState` (serialised as JSON) so that conversations remain coherent across multiple WhatsApp messages.

4. **Reply via API**:
   ```python
   requests.post(
       f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
       headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
       json={"messaging_product": "whatsapp", "to": phone, "text": {"body": agent_reply}}
   )
   ```

5. **Security**: Always verify `X-Hub-Signature-256` on incoming requests to prevent spoofing.

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
