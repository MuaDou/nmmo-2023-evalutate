import glob
import os
import pickle
import queue
import random
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import dateutil.parser
from aicrowd_api import API, AIcrowdSubmission
# from aicrowd_gym.servers.zmq_agent_server import ZmqAgent
from loguru import logger
import numpy as np
# from neurips2022nmmo import TeamResult
from pkg import Mode, util
from pkg.evaluator.evaluator import CompetitionEvaluator
# from pkg.team import AIcrowdAgentTeam
from pkg.evaluator.team_result import FinalTeamResult

CONTENDERS_FILE = "contenders.pkl"
STATE_FILE = "state.pkl"


@dataclass
class Contender:
    id: str
    participant_id: str
    participant_name: str
    grader_id: str
    sr_mu: float = 0.0
    sr_sigma: float = 0.0


# class FinalTeamResult:
#     policy_id: str = None

#     # event-log based, coming from process_event_log
#     total_score: int = 0
#     agent_kill_count: int = 0,
#     npc_kill_count: int = 0,
#     max_progress_to_center: int = 0,
#     eat_food_count: int = 0,
#     drink_water_count: int = 0,
#     item_buy_count: int = 0,

#     # agent object based (fill these in the environment)
#     # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
#     time_alive: int = 0,
#     earned_gold: int = 0,
#     completed_task_count: int = 0,
#     damage_received: int = 0,
#     ration_consumed: int = 0,
#     potion_consumed: int = 0,
    
#     @classmethod
#     def names(cls) -> List[str]:
#         return [
#             "total_score",
#             "agent_kill_count",
#             "npc_kill_count",
#             "max_progress_to_center",
#             "eat_food_count",
#             "drink_water_count",
#             "item_buy_count",
#             "time_alive",
#             "earned_gold",
#             "completed_task_count",
#             "damage_received",
#             "ration_consumed",
#             "potion_consumed",
#         ]

@dataclass
class State:
    n_round: int = 0
    n_match: int = 0
    replays: Dict[str, List[str]] = field(default_factory=lambda: {})

    def collect_replays(self, shared_dir: str, contender_ids: List[str]):
        self.replays = {}
        import glob
        filepaths = glob.glob(
            os.path.join(shared_dir, "replays", f"replay-{Mode.PVP}-*.lzma"))
        for contender_id in contender_ids:
            contender_id = str(contender_id)
            for filepath in filepaths:
                if contender_id in filepath:
                    if contender_id not in self.replays:
                        self.replays[contender_id] = []
                    self.replays[contender_id].append(filepath)

def camelize(x):
    return ''.join(word.title() for word in x.split('_'))

class PVPEvaluator(CompetitionEvaluator):

    def _init_teams(self):
        # self.teams = []

        # self.server.wait_for_agents()
        # agents: Dict[str, ZmqAgent] = self.server.agents
        # for i, agent in enumerate(agents.values()):
        #     if agent.metadata.get("is_user", False):
        #         self.user_team_indices = [i]
        #     self.teams.append(
        #         AIcrowdAgentTeam(agent.metadata["team_id"],
        #                          self.env_config,
        #                          agent,
        #                          policy_id=agent.metadata["policy_id"]))
        # random.shuffle(self.teams)
        pass
    # @staticmethod
    # def load_contenders(shared_dir: str) -> List[Contender]:
    #     path = os.path.join(shared_dir, CONTENDERS_FILE)
    #     if not os.path.exists(path):
    #         return []
    #     with open(path, "rb") as fp:
    #         data = fp.read()
    #     return pickle.loads(data)

    @staticmethod
    def load_contenders(
            shared_dir: str,
            grader_ids: List[str],
            api_token: str,
            deadline: int,
            specific_submissions: List[str] = []) -> List[Contender]:
        q = queue.Queue()
        submission_q = queue.Queue()

        def get_submissions(grader_id: str, thread_id: int):
            api = API(api_token)
            while not q.empty():
                try:
                    submission_id = q.get_nowait()
                except queue.Empty:
                    break

                logger.info(
                    f"[{thread_id}] getting submssion {submission_id} from grader {grader_id}"
                )

                submission = util.guarantee(api.get_submission)(
                    grader_id, submission_id).raw_response
                submission["grader_id"] = grader_id
                submission_q.put(submission)

        api = API(api_token)
        for grader_id in grader_ids:
            submission_ids = api.get_all_submissions(grader_id)
            for i in submission_ids:
                q.put(i)

            threads: List[threading.Thread] = []
            thread_num = int(len(submission_ids) / 5) + 1
            if thread_num > 30:
                thread_num = 30
            for i in range(thread_num):
                threads.append(
                    threading.Thread(target=get_submissions,
                                     args=(grader_id, i),
                                     daemon=True))
                threads[-1].start()

            for thread in threads:
                thread.join()

        logger.info("get submissions done")

        submissions: List[dict] = []
        while not submission_q.empty():
            submissions.append(submission_q.get())

        participant_submissions: Dict[List[dict]] = defaultdict(lambda: [])
        for submission in submissions:
            participant_submissions[submission["participant_id"]].append(
                submission)

        contenders: List[Contender] = []
        for submissions in participant_submissions.values():
            latest = None
            chosen = None
            if specific_submissions:
                for subm in submissions:
                    if str(subm["id"]) in specific_submissions:
                        chosen = subm
                        break
            else:
                for subm in submissions:
                    ts = dateutil.parser.parse(subm["created_at"]).timestamp()
                    if ts > deadline:
                        continue
                    #TODO: cheek this need change
                    ref = subm["meta"].get("repo_ref")
                    if not ref:
                        logger.warning(f"no repo_ref\n{subm}")
                        continue
                    if not ref.endswith("-pvp"):
                        continue
                    if latest is None or (latest is not None and ts > latest):
                        latest = ts
                        chosen = subm

                if not chosen:
                    for subm in submissions:
                        ts = dateutil.parser.parse(
                            subm["created_at"]).timestamp()
                        if ts > deadline:
                            continue
                        if latest is None or (latest is not None
                                              and ts > latest):
                            latest = ts
                            chosen = subm
                else:
                    logger.info(
                        f"choose specific submission for pvp: {chosen}")

            if chosen:
                contenders.append(
                    Contender(str(chosen["id"]), str(chosen["participant_id"]),
                              str(chosen["participant_name"]),
                              chosen["grader_id"]))

        data = pickle.dumps(contenders)
        util.write_data(data, os.path.join(shared_dir, CONTENDERS_FILE), True)

        return contenders

    @staticmethod
    def schedule_matches(
            shared_dir: str,
            grader_ids: List[str],
            api_token: str,
            n_parallel_match: int,
            min_score: float,
            n_round: int,
            deadline: int,
            specific_submissions: List[str] = []) -> List[List[int]]:
        logger.info(f"shared_dir: {shared_dir}")
        logger.info(f"grader_ids: {grader_ids}")
        logger.info(f"n_parallel_match: {n_parallel_match}")
        logger.info(f"min_score: {min_score}")
        logger.info(f"n_round: {n_round}")
        logger.info(f"specific_submissions: {specific_submissions}")
        if os.path.exists(os.path.join(shared_dir, "end")):
            return []

        state = PVPEvaluator.load_state(shared_dir)
        if state.n_round >= n_round:
            return []

        contenders = PVPEvaluator.load_contenders(shared_dir)
        if not contenders:
            logger.info("No contenders found. Generate from AICrowd.")
            contenders = PVPEvaluator.gen_contenders(shared_dir, grader_ids,
                                                     api_token, min_score,
                                                     deadline,
                                                     specific_submissions)
        logger.info(f"contenders: {contenders}")

        state.collect_replays(shared_dir, [c.id for c in contenders])

        matches = []
        contender_ids = [c.id for c in contenders]
        while len(matches) < n_parallel_match:
            random.shuffle(contender_ids)

            left = [x for x in contender_ids]
            n = len(left)
            if n < 16:
                assert n >= 8
                replicas = 2
                n_distinct = 8
            else:
                replicas = 1
                n_distinct = 16
            while len(left) >= n_distinct:
                matches.append(left[:n_distinct] * replicas)
                left = left[n_distinct:]
                if len(matches) >= n_parallel_match:
                    break

        state.n_match += len(matches)
        state.n_round += 1
        PVPEvaluator.save_state(shared_dir, state)

        logger.info(f"Round: {n_round}\nMatches: {matches}")

        return matches

    @staticmethod
    def load_state(shared_dir: str) -> State:
        path = os.path.join(shared_dir, STATE_FILE)
        if not os.path.exists(path):
            return State()
        with open(path, "rb") as fp:
            return pickle.loads(fp.read())

    @staticmethod
    def save_state(shared_dir: str, state: State):
        path = os.path.join(shared_dir, STATE_FILE)
        util.write_data(pickle.dumps(state), path, True)

    @staticmethod
    #TODO:用joseph的代码更新数据
    # rank 用读取的方式来做 等joseph
    def get_final_metrics( history, mode):
        # policy_ids = set()
        # for results in history:
        #     for result_by_team in results.values():
        #         policy_ids.add(result_by_team.policy_id)
                
        # policy_ids = set(history.keys())

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
        #                                 mu=1000,
        #                                 sigma=40)
        # else:
        #     raise Exception(f"invalid mode {mode}")

        # match_count = defaultdict(lambda: 0)
        # # key: policy_id
        # # value: list of result from games
        # all_results: Dict[str, List[TeamResult]] = defaultdict(lambda: [])
        # for result_by_team in history:
        #     teams = []
        #     scores = []
        #     result_by_policy = analyzer.avg_result_by_policy(result_by_team)
        #     for policy_id, result in result_by_policy.items():
        #         teams.append(fake_teams[policy_id])
        #         # use total_score to update true skill
        #         scores.append(result.total_score)
        #         all_results[policy_id].append(result)
        #         match_count[policy_id] += 1
        #     # update true skill
        #     rating_system.update(teams, scores)

        final_metrices: Dict[str, Dict] = {}
        for policy_id, result in history.items():
            # average
            avg_result = defaultdict(lambda: [])
            for name in FinalTeamResult.names():
                avg_result[name].extend([getattr(r, name) for r in result])
            final_metrices[policy_id] = {
                camelize(key): np.mean(avg_result[key])
                for key in avg_result
            }
            #TODO: 用文件读写的方式替代一下
            # final_metrices[policy_id]["TrueSkillMu"] = rating_system.ratings[
            #     policy_id].mu
            # final_metrices[policy_id]["TrueSkillSigma"] = rating_system.ratings[
            #     policy_id].sigma
            # final_metrices[policy_id]["MatchCount"] = match_count[policy_id]
            final_metrices[policy_id]["TrueSkillMu"] = 1
            final_metrices[policy_id]["TrueSkillSigma"] = 0.5
            final_metrices[policy_id]["MatchCount"] = 100

        # if is_pve(mode):
        #     top1_probs = analyzer.topn_prob_by_policy(history)
        #     for policy_id, metrics in final_metrices.items():
        #         final_metrices[policy_id]["Top1Prob"] = top1_probs[policy_id]

        return final_metrices

    @staticmethod
    def evaluate(shared_dir: str,
                 rollout_name: str,
                 api_token: str,
                 local: bool,
                 tournament: str,
                 json_file_path: str,
                 upload_replay: bool = True,
                 dry_run: bool = False,
                 specific_submissions: List[str] = []):
    
        logger.info("Generate contenders from AICrowd.")
        contenders = CompetitionEvaluator.load_contenders(json_file_path)
        print( contenders )
        
        mode = Mode.PVP
        # TODO:重写这个，合并的逻辑
        history = CompetitionEvaluator.load(shared_dir, rollout_name)
        # TODO:生成最终结果的逻辑需要合并
        final = PVPEvaluator.get_final_metrics(history, mode)

        upload_ret = {}
        # if upload_replay:

        #     from pkg import cos
        #     logger.info("upload replay")
        #     upload_ret: dict = util.guarantee(cos.upload_pvp_replays)(
        #         tournament,
        #         os.path.join(shared_dir, "replays"),
        #         list(contenders.keys()),
        #     )

        if dry_run:
            ranked = list(final.keys())
            ranked = sorted(ranked,
                            key=lambda x: final[x]["TrueSkillMu"],
                            reverse=True)
            print(",".join(ranked))
            print("participant_id\tparticipant\tsr\tTotalScore")
            for i, key in enumerate(final):
                p_id, p_name = key.split('_')
                print(
                    f"{p_id}\t{contenders[p_id]['participant_name']}\t{final[key]['TrueSkillMu']:.2f}\t{final[key]['TotalScore']:.2f}"
                )
            return final

        
        api = API(api_token)
        for key, result in final.items():
            p_id = key
            
            logger.info(f"update submission {contenders[p_id]['submit_id']}")
            submission: AIcrowdSubmission = util.guarantee(api.get_submission)(contenders[p_id]['grader_id'], contenders[p_id]['submit_id'])
            submission.api_key = api_token
            submission.message = submission.raw_response["grading_message"]

            # remove outdated PVP-daily*
            if "daily" in tournament:
                meta = {}
                for k, v in submission.meta.itedms():
                    if k.startswith("PVP-daily") or k.startswith("Replay-PVP-daily"):
                        continue
                    meta[k] = v
            else:
                meta = submission.meta

            submission.meta = {
                **meta,
                **{f"PVP-{rollout_name}-{k}": v
                   for k, v in result.items()}
            }
            submission.meta[f"Replay-PVP-{rollout_name}"] = upload_ret.get(
                "submissions", {}).get(p_id, "none")

            if local:
                print(submission.meta)
                # print(submission)
            else:
                submission.grading_status = 'graded'
                submission.score = 1.0
                submission.score_secondary = 1.0
                submission.update(meta_overwrite=True)

        return final

    # def save(self,
    #          shared_dir: str,
    #          history: List[Dict[int, TeamResult]],
    #          mode: Mode,
    #          save_replay: bool,
    #          min_replay: int = 30):
    #     save_replay = False
    #     state = PVPEvaluator.load_state(shared_dir)
    #     if len(state.replays) == len(PVPEvaluator.load_contenders(shared_dir)):
    #         for rs in state.replays.values():
    #             if len(rs) < min_replay:
    #                 save_replay = True
    #                 break
    #     else:
    #         save_replay = True

    #     if save_replay:
    #         logger.info(f"need to save replay")
    #     else:
    #         logger.info(f"no need to save replay")

    #     super().save(shared_dir, history, Mode.PVP, save_replay)
