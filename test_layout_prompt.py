import os
import json

from google import genai
from google.genai import types

from vlmgineer.prompts.layout_prompt_composer import LayoutPromptComposer
from vlmgineer.prompts.schemas.layout_response_schema import LayoutResponseSchema


def main():
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API environment variable")

    client = genai.Client(api_key=api_key)

    composer = LayoutPromptComposer(n_layout_samples=2)
    prompt = composer.create_layout_prompt()

    task_description = """
Task:
Design a simple robotic work-cell layout for a pick-and-place task.

Scene:
A robot must pick a cube from an input area and place it into a target area.
The layout should include:
- one robot base
- one table
- one cube
- one target area
- optionally one obstacle
"""

    full_prompt = prompt.instruction_prompts + "\n" + task_description

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LayoutResponseSchema,
            temperature=0.8,
        ),
    )

    print(response.text)

    parsed = json.loads(response.text)
    print("\nParsed JSON keys:", parsed.keys())


if __name__ == "__main__":
    main()