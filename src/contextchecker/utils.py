"""
Shared utility functions — pure helpers with no side effects.

This file is a leaf dependency — it imports nothing from contextchecker.
"""
import json

def build_compact_schema_example(schema) -> str:
    """Build a compact JSON example from a Pydantic model for vanilla LLM prompts.

    Used as a fallback when the model doesn't support structured output
    and no hand-written vanilla prompt exists in prompt_map.json.
    """
    try:
        json_schema = schema.model_json_schema()
        props = json_schema.get("properties", {})
        example = {k: f"<{v.get('type', 'string')}>" for k, v in props.items()}

        return json.dumps(example, indent=2)
    except Exception:
        return '{"result": "<see prompt for format>"}'


def format_prompt(template: str, variables: dict) -> str:
    """Replace ``{{key}}`` placeholders in *template* with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result
