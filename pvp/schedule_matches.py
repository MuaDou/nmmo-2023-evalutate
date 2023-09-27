from evaluator import AIcrowdEvaluator, Constants
from pkg.evaluator.evaluator import CompetitionEvaluator

AIcrowdEvaluator.folder_check()
contenders = AIcrowdEvaluator.gen_contenders()
CompetitionEvaluator.save_contenders( contenders, Constants.CONTENDER_JSON_PATH )
# with open("/tmp/next-matches", "w") as fp:
#     fp.write(json.dumps(matches))