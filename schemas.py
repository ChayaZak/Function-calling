
FINAL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step explanation of what you did and why"
        },
        "tools_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names used"
        },
        "final_answer": {
            "type": "string",
            "description": "The complete, user-friendly answer to the question"
        }
    },
    "required": ["reasoning", "tools_used", "final_answer"],
}

AGENT_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Returns the current weather for a given city.",
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
        "name": "get_current_time",
        "description": "Returns the current local time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    },
    {
        "type": "function",
        "name": "calculator",
        "description": "Evaluates a safe mathematical expression and returns the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '2 + 2' or '15 * 4 / 2'"
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        }
    }
]
