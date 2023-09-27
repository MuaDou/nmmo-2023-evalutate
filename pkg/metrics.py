import numpy as np
from collections import defaultdict
from typing import Dict, List
# from evaluator.team_result import FinalTeamResult
# from neurips2022nmmo import Team, CompetitionConfig, RatingSystem, analyzer, TeamResult

from pkg import Mode


def camelize(x):
    return ''.join(word.title() for word in x.split('_'))


def is_pve(mode: Mode) -> bool:
    return mode in [
        Mode.PVE_STAGE1, Mode.PVE_STAGE2, Mode.PVE_STAGE3, Mode.PVE_BONUS
    ]


def is_pvp(mode: Mode) -> bool:
    return mode == Mode.PVP


def get_final_metrices(history: List[Dict[int, FinalTeamResult]],
                       mode: Mode) -> Dict[str, Dict[str, float]]:
    policy_ids = set()
    for results in history:
        for result_by_team in results.values():
            policy_ids.add(result_by_team.policy_id)

    # fake_teams = {
    #     policy_id: Team(policy_id, CompetitionConfig())
    #     for policy_id in policy_ids
    # }
    # if is_pve(mode):
    #     rating_system = RatingSystem(
    #         teams=list(fake_teams.values()),
    #         mu=1000,
    #         sigma=400,
    #     )
    # elif is_pvp(mode):
    #     rating_system = RatingSystem(list(fake_teams.values()),
    #                                  mu=1000,
    #                                  sigma=40)
    # else:
    #     raise Exception(f"invalid mode {mode}")

    match_count = defaultdict(lambda: 0)
    # key: policy_id
    # value: list of result from games
    all_results: Dict[str, List[TeamResult]] = defaultdict(lambda: [])
    for result_by_team in history:
        teams = []
        scores = []
        result_by_policy = analyzer.avg_result_by_policy(result_by_team)
        for policy_id, result in result_by_policy.items():
            teams.append(fake_teams[policy_id])
            # use total_score to update true skill
            scores.append(result.total_score)
            all_results[policy_id].append(result)
            match_count[policy_id] += 1
        # update true skill
        rating_system.update(teams, scores)

    final_metrices: Dict[str, Dict] = {}
    for policy_id, result in all_results.items():
        # average
        avg_result = defaultdict(lambda: [])
        for name in TeamResult.names():
            avg_result[name].extend([getattr(r, name) for r in result])
        final_metrices[policy_id] = {
            camelize(key): np.mean(avg_result[key])
            for key in avg_result
        }
        final_metrices[policy_id]["TrueSkillMu"] = rating_system.ratings[
            policy_id].mu
        final_metrices[policy_id]["TrueSkillSigma"] = rating_system.ratings[
            policy_id].sigma
        final_metrices[policy_id]["MatchCount"] = match_count[policy_id]

    if is_pve(mode):
        top1_probs = analyzer.topn_prob_by_policy(history)
        for policy_id, metrics in final_metrices.items():
            final_metrices[policy_id]["Top1Prob"] = top1_probs[policy_id]

    return final_metrices
