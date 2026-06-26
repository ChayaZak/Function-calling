import json
import os

from datetime import datetime
from typing import Any, Callable

from google import genai
from google.genai import types

from dotenv import load_dotenv

from schemas import AGENT_TOOLS, FINAL_OUTPUT_SCHEMA

MODEL = "gemini-2.5-flash-lite"
QUESTION = "What is the weather in Tel Aviv right now, and what time is it?"

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
    "new york": { "city": "new york", "temperature_C":22, "condition": "Rain"},
    "paris": { "city": "paris", "temperature_C": 19, "condition": "Clear"},
    "berlin": {"city": "berlin",  "temperature_C": 11, "condition": "Overcast"}
}


ToolFunction = Callable[..., dict[str, Any]]

def get_weather(city: str) -> dict[str, Any]:
    city_key= city.strip().lower()
    weather = WEATHER.get(city_key)
    if weather is None:
        return{
            "found": False,
            "city": city,
            "message": "City not found"
        }
    return { "found": True, **weather }

def get_current_time() -> dict[str, Any]:
    now = datetime.now()
    return {
        "found": True,
        "time": now.strftime("%H:%M"),
        "timezone": "IDT",
    }

def call_tool(name: str, argument: dict[str, Any], available_tools: dict[str, ToolFunction]) -> dict[str, Any]:
    tool_function = available_tools.get(name)
    if tool_function is None:
        return {"error": f"Tool {name} not found"}
    return tool_function(**argument)

def ask_llm(client: genai.Client, model: str, history: list[types.Content])-> types.GenerateContentResponse:
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
    response = client.models.generate_content(
        model=model,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[sdk_tools],
        ),
    )
    return response

def run_demo(client: genai.Client, model: str, question: str) -> str:
    available_tools: dict[str, ToolFunction] = {
        "get_weather": get_weather,
        "get_current_time": get_current_time
    }
    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    for round_number in range(1, 6):
        print(f"[round {round_number}] calling LLM …")
        response = ask_llm(client, model, history)

        if not response.candidates:
            return "No candidates returned from model."

        candidate = response.candidates[0]
        content = candidate.content
        if content is None:
            return "Empty content returned from model."

        history.append(content)

        tool_call_parts = [
            part for part in content.parts
            if part.function_call is not None
        ]

        if not tool_call_parts:
            print(f"[round {round_number}] no tool calls — requesting final answer …")
            raw = response.text

            if not raw:
                return "Model returned empty final response."
            try:
                parsed = json.loads(raw)
                return parsed.get("final_answer", raw)
            except json.JSONDecodeError:
                return raw

        tool_response_parts: list[types.Part] = []
        for part in tool_call_parts:
            fc = part.function_call
            arguments = dict(fc.args) if fc.args else {}
            print(f"  → tool call : {fc.name}({arguments})")
            result = call_tool(fc.name, arguments, available_tools)
            print(f"  ← tool result: {result}")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=result,
                    )
                )
            )

        history.append(
            types.Content(role="user", parts=tool_response_parts)
        )

    return "Max rounds reached without a final answer."


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    print(QUESTION)

    try:
        final_answer = run_demo(client, MODEL, QUESTION)
        print(final_answer)
        return
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()

