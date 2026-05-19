import json
import glob
import os

from vlmgineer.evaluator.pybullet_layout_evaluator import PyBulletLayoutEvaluator


def find_latest_layout_results():
    files = glob.glob("layout_results_*.json")

    if not files:
        raise FileNotFoundError("No layout_results_*.json file found.")

    return max(files, key=os.path.getmtime)


def main():
    results_file = find_latest_layout_results()
    print(f"Loading: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    evaluator = PyBulletLayoutEvaluator(gui=False)

    print("\n=== PYBULLET PHYSICAL EVALUATION ===")

    for layout_item in data["evaluated_layouts"]:
        layout = layout_item["layout"]
        result = evaluator.evaluate_layout(layout)

        print(f"\nLayout: {layout['layout_name']}")
        print(f"physical_feasibility_score: {result['physical_feasibility_score']:.3f}")
        print(f"combined_physical_score:    {result['combined_physical_score']:.3f}")
        print(f"robot_table_clearance:      {result['robot_table_clearance']}")
        print(f"ik_reachability_score:      {result['ik_reachability_score']:.3f}")
        print(f"collision_free_score:       {result['collision_free_score']:.3f}")

        for detail in result["details"]:
            print(
                f"  - {detail['object_name']} | "
                f"reachable={detail['reachable']} | "
                f"collision_free={detail['collision_free']} | "
                f"physically_feasible={detail['physically_feasible']} | "
                f"trajectory_collision={detail['trajectory_collision']} |"
                f"safety_margin_ok={detail['safety_margin']['safety_margin_ok']}"
            )

            if not detail["safety_margin"]["safety_margin_ok"]:
                print("    safety margin violations:")
                for obj in detail["safety_margin"]["too_close_objects"]:
                    print(
                        f"      * {obj['object_name']} "
                        f"[{obj['object_type']}] "
                        f"min_distance={obj['min_distance']:.4f}"
                    )

            if detail["forbidden_collisions"]:
                print("    forbidden collisions:")
                for col in detail["forbidden_collisions"]:
                    print(
                        f"      * {col['object_name']} "
                        f"[{col['object_type']}] "
                        f"({col['num_contacts']} contacts)"
                    )


if __name__ == "__main__":
    main()