from evaluator import AIcrowdEvaluator


evaluator = AIcrowdEvaluator()
evaluator.setup_pufferlib()
evaluator.serve()

