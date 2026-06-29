import ast
import datetime
import json
import operator
import os
from typing import Any, Callable

from google import genai
from google.genai import types

from dotenv import load_dotenv

from schemas import AGENT_TOOLS, FINAL_OUTPUT_SCHEMA

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

ToolFunction = Callable[..., dict[str, Any]]


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


def call_tool(
    name: str,
    argument: dict[str, Any],
    available_tools: dict[str, ToolFunction],
) -> dict[str, Any]:
    tool_function = available_tools.get(name)
    if tool_function is None:
        return {"error": f"Tool '{name}' not found"}
    return tool_function(**argument)


def ask_llm(
    client: genai.Client, model: str, history: list[types.Content]
) -> types.GenerateContentResponse:
    sdk_tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters_json_schema=tool.get("parameters"),
            )
            for tool in AGENT_TOOLS
        ]
    )

    # response_schema is intentionally omitted: setting it alongside tools causes
    # Gemini to skip function calls and return JSON directly, breaking the loop.
    # The SYSTEM_PROMPT enforces the JSON structure for the final text response.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[sdk_tools],
    )

    return client.models.generate_content(
        model=model,
        contents=history,
        config=config,
    )


def _validate_schema(data: dict) -> list[str]:
    return [k for k in FINAL_OUTPUT_SCHEMA.get("required", []) if k not in data]


def run_demo(client: genai.Client, model: str, question: str) -> dict | str:
    available_tools: dict[str, ToolFunction] = {
        "get_weather": get_weather,
        "get_current_time": get_current_time,
        "calculator": calculator,
    }
    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    for round_number in range(1, 6):
        print(f"\n[round {round_number}] Calling LLM ...")
        response = ask_llm(client, model, history)

        if not response.candidates:
            return "No candidates returned."

        content = response.candidates[0].content
        history.append(content)

        tool_call_parts = [p for p in content.parts if p.function_call]

        if not tool_call_parts:
            print(f"[round {round_number}] No more tool calls — parsing final answer ...")
            raw = response.text or ""
            clean_json = raw.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(clean_json)
            except json.JSONDecodeError:
                return {"error": "Model returned invalid JSON", "raw": raw}

            missing = _validate_schema(result)
            if missing:
                print(f"  [warn] Final JSON is missing required keys: {missing}")

            print(f"  [final] reasoning   : {str(result.get('reasoning', ''))[:120]}")
            print(f"  [final] tools_used  : {result.get('tools_used', [])}")
            print(f"  [final] final_answer: {result.get('final_answer', '')}")
            return result

        tool_response_parts = []
        for part in tool_call_parts:
            fc = part.function_call
            args = dict(fc.args)
            print(f"  [tool call  ] {fc.name}({args})")
            tool_result = call_tool(fc.name, args, available_tools)
            print(f"  [tool result] {tool_result}")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name, response=tool_result
                    )
                )
            )

        history.append(types.Content(role="user", parts=tool_response_parts))

    return "Max rounds reached without a final answer."


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)
    print(f"Question: {QUESTION}\n")

    try:
        final_answer = run_demo(client, MODEL, QUESTION)
        print("\n=== Final Answer ===")
        print(final_answer.get("final_answer", ""))
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
