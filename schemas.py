# schemas.py

# סכמה עבור התשובה הסופית של המודל
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
    "additionalProperties": False
}

# רשימת הכלים (Tools) שהסוכן מכיר
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
            "properties": {}, # אין צורך בפרמטרים לפונקציה הזו
            "additionalProperties": False,
        }
    }
]