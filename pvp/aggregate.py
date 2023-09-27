import json
from evaluator import AIcrowdEvaluator

ev = AIcrowdEvaluator()
final = ev.aggregate_results()
print(json.dumps(final, indent=2))

