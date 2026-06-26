import json
import os
from typing import Any, Callable

from google import genai
from google.genai import types

from dotenv import load_dotenv


MODEL = "gemini-2.5-flash"
QUESTION = "What is the weather in Tel Aviv? "
# QUESTION = "2+9"




SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use the tools to gather all the information needed to answer the user.

When you have finished all tool calls and are ready to give the final answer,
respond ONLY with a valid JSON object — no markdown fences, no extra prose.
Use this exact schema (field order matters — reasoning must come first):

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

TOOLS = [
    {
    "type": "function",
    "name": "get_weather",
    "description": "Returns the current weather for a given city.",
    "script": True,
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'Tel Aviv' or 'London'"
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    }
    },
    {
        "type": "function",
        "name": "calculator",
        "description": "Returns the calculate answer for given numbers.",
        "script": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '15 * 7 + 3'"
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        }
    }
]

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

def calculator(expression: str) -> dict[str, Any]:
    try:
        result = eval(expression)
        return {"found": True, "result": result}
    except Exception as error:
        return {"found": False, "error": str(error)}

def call_tool(name: str, argument: dict[str, Any], available_tools: dict[str, ToolFunction]) -> dict[str, Any]:
    tool_function = available_tools.get(name)
    if tool_function is None:
        return {"error": f"Tool {name} not found"}
    try:
        return tool_function(**argument)
    except Exception as error:
        return {"error": f"Tool execution failed: {error}"}

def ask_llm(client: genai.Client, model: str, history: list[types.Content])-> types.GenerateContentResponse:
    sdk_tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters_json_schema=tool.get("parameters"),
            )
            for tool in TOOLS
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
    print(question)
    available_tools: dict[str, ToolFunction] = {"get_weather": get_weather, "calculator": calculator}

    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    for round_number in range(1, 6):
        print(f"[round {round_number}] calling LLM …")
        response = ask_llm(client, model, history)

        # candidates and content can be None — guard defensively
        if not response.candidates:
            return "No candidates returned from model."

        candidate = response.candidates[0]
        content = candidate.content
        if content is None:
            return "Empty content returned from model."

        # Append model turn to history
        history.append(content)

        # Collect function-call parts
        tool_call_parts = [
            part for part in content.parts
            if part.function_call is not None
        ]

        # No tool calls → model is done; extract JSON final answer
        if not tool_call_parts:
            raw_text = "".join(
                part.text for part in content.parts if part.text
            ).strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            try:
                parsed = json.loads(raw_text)
                return parsed.get("final_answer", raw_text)
            except json.JSONDecodeError:
                return raw_text

        # Execute tool calls and feed results back as role="user"
        # (the SDK only accepts "user" or "model" as Content roles;
        #  function responses must come from the "user" side)
        tool_response_parts: list[types.Part] = []
        for part in tool_call_parts:
            fc = part.function_call
            arguments = dict(fc.args)
            print(f"  → tool call : {fc.name}({arguments})")
            result = call_tool(fc.name, arguments, available_tools)
            print(f"  ← tool result: {result}")
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response=result,   # dict, not JSON string
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
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)

    try:
        final_answer = run_demo(client, MODEL, QUESTION)
        print(final_answer)
        return
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()


