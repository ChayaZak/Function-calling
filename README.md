# LLM Agent with Tool Calling & Structured Output

A minimal but complete **agentic loop** built on the **Google Gemini 2.5 Flash** API.  
The agent receives a natural-language question, decides which tools to call, executes them, and returns a validated structured JSON answer — all in an autonomous multi-round loop.

---

## What it does

```
User question
     │
     ▼
┌─────────────────────────────────────────────────┐
│                  Agent Loop                     │
│                                                 │
│  ┌──────────┐    tool calls    ┌─────────────┐  │
│  │  Gemini  │ ──────────────▶ │  Tool Layer │  │
│  │ 2.5 Flash│ ◀────────────── │  (Python)   │  │
│  └──────────┘   tool results  └─────────────┘  │
│       │                                         │
│       │  (no more tool calls)                   │
│       ▼                                         │
│  Structured JSON answer (schema-validated)      │
└─────────────────────────────────────────────────┘
```

The model keeps calling tools until it has all the information it needs, then produces a final answer that is parsed and validated against a defined JSON schema.

---

## Features

- **Multi-round agentic loop** — the LLM can call multiple tools across several rounds before answering
- **3 built-in tools** — weather lookup, current time, and a safe expression calculator
- **Structured output** — final response is always parsed and validated against a JSON schema
- **Safe calculator** — uses Python's `ast` module instead of `eval()`, allowing only numeric operations
- **Full iteration logging** — every round, every tool call, and every result is printed
- **Clean architecture** — tools, schemas, and agent logic are cleanly separated

---

## Example output

```
Question: What is the weather in Tel Aviv right now, what time is it, and what is 15 * 4 + 7?

[round 1] Calling LLM ...
  [tool call  ] get_weather({'city': 'Tel Aviv'})
  [tool result] {'found': True, 'city': 'tel aviv', 'temperature_C': 28, 'condition': 'Sunny'}
  [tool call  ] get_current_time({})
  [tool result] {'found': True, 'time': '14:32', 'timezone': 'IDT'}
  [tool call  ] calculator({'expression': '15 * 4 + 7'})
  [tool result] {'found': True, 'expression': '15 * 4 + 7', 'result': 67}

[round 2] Calling LLM ...
[round 2] No more tool calls — parsing final answer ...
  [final] reasoning   : I called get_weather to get Tel Aviv's weather, get_current_time for the local time, and calculator for the math.
  [final] tools_used  : ['get_weather', 'get_current_time', 'calculator']
  [final] final_answer: It is currently 14:32 IDT. The weather in Tel Aviv is Sunny at 28°C. 15 * 4 + 7 = 67.

=== Final Answer ===
It is currently 14:32 IDT. The weather in Tel Aviv is Sunny at 28°C. 15 * 4 + 7 = 67.
```

---

## Project structure

```
Function-calling/
├── Function_calling.py   # Entry point — agent loop, tool implementations
├── schemas.py            # Tool definitions (AGENT_TOOLS) and output schema
├── .env                  # API key (not committed)
└── README.md
```

| File | Responsibility |
|------|----------------|
| `schemas.py` | Declares all tool schemas (`AGENT_TOOLS`) and the `FINAL_OUTPUT_SCHEMA` the model must conform to |
| `Function_calling.py` | Implements the tools, the `ask_llm` round, the agentic loop (`run_demo`), and `main` |

---

## Setup

**Prerequisites:** Python 3.12+

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Function-calling
pip install google-genai python-dotenv
```

### 2. Set your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 3. Run

```bash
python Function_calling.py
```

---

## How the agent loop works

1. The user's question is added to the conversation `history` as a `user` turn.
2. The full history is sent to Gemini with the tool definitions attached.
3. **If the model returns function calls** → each tool is executed locally and the results are appended to the history as a `user` turn; the loop repeats.
4. **If the model returns text** → the response is parsed as JSON and validated against `FINAL_OUTPUT_SCHEMA`.
5. The loop caps at 5 rounds to prevent infinite execution.

> **Key design decision:** `response_schema` is intentionally *not* set in the `GenerateContentConfig` alongside `tools`. When both are present, Gemini's constrained-decoding mode takes priority and the model skips tool calls entirely. The system prompt enforces the JSON structure instead, and the schema is used only for post-hoc validation.

---

## Available tools

| Tool | Input | Returns |
|------|-------|---------|
| `get_weather(city)` | City name (string) | Temperature (°C) and weather condition |
| `get_current_time()` | — | Current time (HH:MM) and timezone |
| `calculator(expression)` | Math expression string | Numeric result |

### Adding a new tool

**Step 1** — add the schema to `AGENT_TOOLS` in `schemas.py`:

```python
{
    "type": "function",
    "name": "my_tool",
    "description": "What this tool does.",
    "parameters": {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"],
        "additionalProperties": False,
    }
}
```

**Step 2** — implement the function in `Function_calling.py` and register it:

```python
def my_tool(param: str) -> dict[str, Any]:
    return {"found": True, "result": ...}

# inside run_demo:
available_tools = {
    ...,
    "my_tool": my_tool,
}
```

That's it — the agent loop picks it up automatically.

---

## Tech stack

| | |
|---|---|
| **LLM** | Google Gemini 2.5 Flash |
| **SDK** | `google-genai` |
| **Language** | Python 3.12 |
| **Key concepts** | Tool / function calling, agentic loop, structured output, safe AST evaluation |
