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
LLM_PROVIDER=anthropic            # or: openai | google
ANTHROPIC_API_KEY=sk-ant-...      # Claude 3 Haiku (default)
# OPENAI_API_KEY=sk-...           # GPT-4o-mini (alternative)
# GOOGLE_API_KEY=AIza...          # Gemini 1.5 Flash (alternative)
```

### 5. Run the agent

```bash
python main.py
```

You'll see an interactive CLI. Type your messages and press Enter. Type `quit` to exit.

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

The agent is built as a **directed acyclic graph** with four nodes:

1. **`intent_detector`** — Every user message first passes through an LLM-powered intent classifier that labels it as `greeting`, `product_inquiry`, or `high_intent`.
2. **`greeter`** — Handles casual conversation using a system-prompted LLM response.
3. **`rag_answer`** — Performs keyword-based RAG retrieval from `autostream_kb.json`, injects the context into the LLM prompt, and returns an accurate answer.
4. **`lead_capture`** — A **multi-step state machine** (sub-FSM inside the node) that advances through `collecting_name → collecting_email → collecting_platform → done`, only calling `mock_lead_capture()` once all three fields are validated and collected.

### State Management

The full `AgentState` TypedDict is passed through and returned from every node, ensuring **persistent memory across 5–6 conversation turns** with no loss of context. The `messages` list (using LangGraph's `add_messages` reducer) accumulates the complete conversation history, which is replayed to the LLM on each turn for coherent multi-turn dialogue.

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
| Intent detection | LLM classifier node with 3-label output |
| RAG | Keyword-based retrieval from `autostream_kb.json` |
| State management | `AgentState` TypedDict persisted across turns via LangGraph |
| Tool calling logic | `mock_lead_capture()` only fires after all 3 fields are collected |
| Code clarity | Modular structure: `agent/`, `tools/`, `knowledge_base/` |
| Deployability | WhatsApp webhook architecture documented above |

---

##  Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.9+ | Requirement |
| Framework | LangGraph | Deterministic state graph, superior state management |
| LLM | Claude 3 Haiku (default) | Fast, cost-efficient, instruction-following |
| Knowledge Base | Local JSON | Simple, no external vector DB needed for this scale |
| State | `AgentState` TypedDict | Full conversation + lead data persisted across turns |

---

##  License

MIT — built as a technical assignment for ServiceHive / Inflx.
