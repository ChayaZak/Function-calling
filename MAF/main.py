import json
import os
import sys
import datetime
import ast
import operator
from typing import Any, Callable

from dotenv import load_dotenv
import autogen

# Allow importing schemas.py from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import FINAL_OUTPUT_SCHEMA

MODEL = "gemini-2.5-flash"
QUESTION = "What is the weather in Tel Aviv right now, what time is it, and what is 15 * 4 + 7?"

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Rules:
- You MUST call the relevant tool(s) before answering — never answer from memory.
- Every fact in your answer must come from a tool result.
- Call all needed tools first, then give the final answer.

When all tool calls are done, respond ONLY with this exact JSON (no markdown, no extra text):

{
  "reasoning": "<step-by-step explanation of what you did and why>",
  "tools_used": ["<tool_name>", ...],
  "final_answer": "<complete, user-friendly answer to the question>"
}
"""

WEATHER = {
    "tel aviv": {"city": "tel aviv", "temperature_C": 28, "condition": "Sunny"},
    "london": {"city": "london", "temperature_C": 14, "condition": "Cloudy"},
    "new york": {"city": "new york", "temperature_C": 22, "condition": "Rain"},
    "paris": {"city": "paris", "temperature_C": 19, "condition": "Clear"},
    "berlin": {"city": "berlin", "temperature_C": 11, "condition": "Overcast"},
}


def get_weather(city: str) -> dict[str, Any]:
    city_key = city.strip().lower()
    weather = WEATHER.get(city_key)
    if weather is None:
        return {"found": False, "city": city, "message": "City not found"}
    return {"found": True, **weather}


def get_current_time() -> dict[str, Any]:
    now = datetime.datetime.now()
    return {"found": True, "time": now.strftime("%H:%M"), "timezone": "IDT"}


_CALC_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _CALC_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _CALC_OPS[op](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _CALC_OPS:
            raise ValueError(f"Unsupported unary operator: {op.__name__}")
        return _CALC_OPS[op](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculator(expression: str) -> dict[str, Any]:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return {"found": True, "expression": expression, "result": result}
    except Exception as e:
        return {"found": False, "expression": expression, "error": str(e)}


def _is_final_answer(msg: dict) -> bool:
    """Terminate once the assistant sends valid JSON with all required schema keys."""
    content = msg.get("content") or ""
    clean = content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        return all(k in data for k in FINAL_OUTPUT_SCHEMA["required"])
    except (json.JSONDecodeError, AttributeError):
        return False


def _parse_final(content: str) -> dict:
    clean = content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def _validate_schema(data: dict) -> list[str]:
    return [k for k in FINAL_OUTPUT_SCHEMA.get("required", []) if k not in data]


def main():
    load_dotenv()

    llm_config = {
        "config_list": [{
            "model": MODEL,
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "api_type": "google",
        }]
    }

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=SYSTEM_PROMPT,
    )

    # MAF hides what Function_calling.py does manually: the tool-call routing,
    # conversation-state accumulation, and the round loop are all handled inside
    # initiate_chat(); is_termination_msg replaces the explicit `for round` check.
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,  # tools only — no Python code execution
        max_consecutive_auto_reply=10,
        is_termination_msg=_is_final_answer,
    )

    autogen.register_function(
        get_weather, caller=assistant, executor=user_proxy,
        name="get_weather", description="Returns current weather for a given city.",
    )
    autogen.register_function(
        get_current_time, caller=assistant, executor=user_proxy,
        name="get_current_time", description="Returns the current local time.",
    )
    autogen.register_function(
        calculator, caller=assistant, executor=user_proxy,
        name="calculator", description="Evaluates a safe mathematical expression.",
    )

    print(f"Question: {QUESTION}\n")

    user_proxy.initiate_chat(assistant, message=QUESTION)

    # Extract the final structured answer from the assistant's message history.
    # In user_proxy.chat_messages[assistant], AutoGen labels user_proxy's own
    # messages as role="assistant" (reversed perspective), so we read from the
    # assistant's side where its own messages are correctly role="assistant".
    messages = assistant.chat_messages.get(user_proxy, [])
    final_content = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls")),
        None,
    )

    if not final_content:
        print("Error: No final answer found in chat history.")
        return

    try:
        result = _parse_final(final_content)
    except json.JSONDecodeError:
        print(f"Error: Model returned invalid JSON.\nRaw: {final_content}")
        return

    missing = _validate_schema(result)
    if missing:
        print(f"  [warn] Final JSON is missing required keys: {missing}")

    print(f"\n  [final] reasoning   : {str(result.get('reasoning', ''))[:120]}")
    print(f"  [final] tools_used  : {result.get('tools_used', [])}")
    print(f"  [final] final_answer: {result.get('final_answer', '')}")

    print("\n=== Final Answer ===")
    print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
