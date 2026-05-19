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
        model="gemini-3.1-flash-lite",
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
            0.3 * semantic_score
            + 0.7 * physical_score
        )

        if physical_metrics["physical_feasibility_score"] == 0.0:
            final_score *= 0.5

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

        print("Target details:")

        for detail in physical["details"]:
            print(
                f"  - {detail['object_name']} | "
                f"reachable={detail['reachable']} | "
                f"collision_free={detail['collision_free']} | "
                f"trajectory_collision={detail['trajectory_collision']} | "
                f"safety_margin_ok={detail['safety_margin']['safety_margin_ok']} | "
                f"physically_feasible={detail['physically_feasible']}"
            )

            if not detail["safety_margin"]["safety_margin_ok"]:
                print("    safety margin violations:")
                for obj in detail["safety_margin"]["too_close_objects"]:
                    print(
                        f"      * {obj['object_name']} "
                        f"[{obj['object_type']}] "
                        f"min_distance={obj['min_distance']:.4f}"
                    )

    best = evaluated_layouts[0]

    print("\n=== BEST FINAL LAYOUT ===")
    print(best["layout_name"])
    print(f"Final score: {best['final_score']:.3f}")
    print(best["layout_description"])


def main():
    max_attempts = 3

    best_overall_layout_data = None
    best_overall_evaluated_layouts = None
    best_overall_score = -1.0

    for attempt in range(1, max_attempts + 1):
        print(f"\n==============================")
        print(f"GENERATION ATTEMPT {attempt}/{max_attempts}")
        print(f"==============================")

        print("Generating layouts with Gemini...")
        layout_data = generate_layouts()

        print("Evaluating generated layouts...")
        evaluated_layouts = evaluate_layouts(layout_data)

        best_layout = evaluated_layouts[0]
        best_score = best_layout["final_score"]
        best_physical_feasibility = best_layout["physical_metrics"]["physical_feasibility_score"]

        print_ranking(evaluated_layouts)

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_layout_data = layout_data
            best_overall_evaluated_layouts = evaluated_layouts

        if best_physical_feasibility == 1.0:
            print("\nFully physically feasible layout found. Stopping optimization loop.")
            break
        else:
            print("\nNo fully physically feasible layout found in this attempt.")
            print("Trying another generation batch...")

    print("\n==============================")
    print("BEST OVERALL RESULT")
    print("==============================")

    print_ranking(best_overall_evaluated_layouts)

    output_path = save_results(
        best_overall_layout_data,
        best_overall_evaluated_layouts,
    )

    print(f"\nBest overall results saved to: {output_path}")


if __name__ == "__main__":
    main()