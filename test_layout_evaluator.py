import json

from vlmgineer.evaluator.layout_evaluator import LayoutEvaluator


with open("layout_sample.json", "r") as f:
    data = json.load(f)

evaluator = LayoutEvaluator()

for layout in data["layout_strategies"]:
    result = evaluator.evaluate_layout(layout)
    print("\nLayout:", layout["layout_name"])
    print(result)