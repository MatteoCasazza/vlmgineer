import os
import json
from datetime import datetime

from google import genai
from google.genai import types

from vlmgineer.prompts.layout_prompt_composer import LayoutPromptComposer
from vlmgineer.prompts.schemas.layout_response_schema import LayoutResponseSchema
from vlmgineer.evaluator.layout_evaluator import LayoutEvaluator
from vlmgineer.evaluator.pybullet_layout_evaluator import PyBulletLayoutEvaluator


def generate_layouts():
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API environment variable")

    client = genai.Client(api_key=api_key)

    composer = LayoutPromptComposer(n_layout_samples=4)
    prompt = composer.create_layout_prompt()

    task_description = """
Task:
Design a robotic work-cell layout for a pick-and-place task.

Scene:
A robot must pick a cube from an input area and place it into a target area.

The work-cell should include:
- one robot base
- one table
- one cube
- one target area
- optionally one obstacle

Design diverse candidate layouts. Some layouts can be simple and efficient, while others can test more constrained or obstacle-aware configurations.
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

    return json.loads(response.text)


def evaluate_layouts(layout_data):
    layout_evaluator = LayoutEvaluator()
    pybullet_evaluator = PyBulletLayoutEvaluator(gui=False)

    evaluated_layouts = []

    for layout in layout_data["layout_strategies"]:
        semantic_metrics = layout_evaluator.evaluate_layout(layout)
        physical_metrics = pybullet_evaluator.evaluate_layout(layout)

        semantic_score = semantic_metrics["layout_score"]
        physical_score = physical_metrics["combined_physical_score"]

        final_score = (
            0.4 * semantic_score
            + 0.6 * physical_score
        )

        evaluated_layout = {
            "layout_name": layout["layout_name"],
            "layout_description": layout["layout_description"],
            "layout": layout,
            "semantic_metrics": semantic_metrics,
            "physical_metrics": physical_metrics,
            "final_score": final_score,
        }

        evaluated_layouts.append(evaluated_layout)

    evaluated_layouts.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return evaluated_layouts


def save_results(layout_data, evaluated_layouts):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": timestamp,
        "scene_analysis": layout_data["scene_analysis"],
        "task_analysis": layout_data["task_analysis"],
        "evaluated_layouts": evaluated_layouts,
        "best_layout": evaluated_layouts[0],
    }

    output_path = f"layout_results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def print_ranking(evaluated_layouts):
    print("\n=== FINAL LAYOUT RANKING ===")

    for idx, item in enumerate(evaluated_layouts, start=1):
        semantic = item["semantic_metrics"]
        physical = item["physical_metrics"]

        print(f"\n#{idx} - {item['layout_name']}")
        print(f"Final score:              {item['final_score']:.3f}")
        print(f"Semantic layout score:    {semantic['layout_score']:.3f}")
        print(f"Physical combined score:  {physical['combined_physical_score']:.3f}")
        print(f"IK reachability score:    {physical['ik_reachability_score']:.3f}")
        print(f"Collision-free score:     {physical['collision_free_score']:.3f}")
        print(f"Physical feasibility:     {physical['physical_feasibility_score']:.3f}")
        print(f"Robot-table clearance:    {physical['robot_table_clearance']}")

    best = evaluated_layouts[0]

    print("\n=== BEST FINAL LAYOUT ===")
    print(best["layout_name"])
    print(f"Final score: {best['final_score']:.3f}")
    print(best["layout_description"])


def main():
    print("Generating layouts with Gemini...")
    layout_data = generate_layouts()

    print("Evaluating generated layouts...")
    evaluated_layouts = evaluate_layouts(layout_data)

    print_ranking(evaluated_layouts)

    output_path = save_results(layout_data, evaluated_layouts)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()