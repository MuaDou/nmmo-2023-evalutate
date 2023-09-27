import os
import json
import multiprocessing
from loguru import logger
from typing import List, Dict
from neurips2022nmmo import Team, TeamResult, analyzer
from aicrowd_gym.servers.zmq_agent_server import ZmqAgent
from aicrowd_gym.clients.zmq_oracle_client import ZmqOracleClient

from pkg import util
from pkg import metrics
from pkg import markdown
from pkg.mode import Mode
from pkg.team import AIcrowdAgentTeam
from pkg.serializer import PickleSerializer
from pkg.evaluator.evaluator import CompetitionEvaluator


class PVEEvaluator(CompetitionEvaluator):
    ps: List[multiprocessing.Process]

    pass_values: Dict[Mode, float] = {}

    def _start_baseline_teams(self) -> List[multiprocessing.Process]:
        raise NotImplementedError

    def _start_teams(self, teams: List[Team]) -> List[multiprocessing.Process]:

        def _start(team: Team):
            import os
            import numpy as np

            # TODO: ensure init done
            # the aicrowd_gym/clients/zmq_oracle_client.py:L53 should be capable of
            # selecting an unused port
            np.random.seed(os.getpid())
            client = None
            while 1:
                try:
                    client = ZmqOracleClient(self.host, self.port,
                                             PickleSerializer())
                except:
                    logger.exception("init zmq oracle client failed")
                else:
                    break
            client.register_agent(agent=team,
                                  metadata={
                                      "team_id": team.id,
                                      "is_user": False,
                                      "policy_id": team.policy_id
                                  })
            client.run_agent()

        if multiprocessing.get_start_method() != "fork":
            multiprocessing.set_start_method("fork", force=True)
        ps = []
        for team in teams:
            p = multiprocessing.Process(target=_start,
                                        args=(team, ),
                                        daemon=True)
            p.start()
            ps.append(p)

        return ps

    def _init_teams(self):
        self.teams = []

        self.ps = self._start_baseline_teams()

        self.server.wait_for_agents()
        agents: Dict[str, ZmqAgent] = self.server.agents
        for i, agent in enumerate(agents.values()):
            if agent.metadata.get("is_user", False):
                self.user_team_indices = [i]
                self.teams.append(
                    AIcrowdAgentTeam(agent.metadata["team_id"],
                                     self.env_config,
                                     agent,
                                     policy_id=agent.metadata["policy_id"]))
                logger.info(
                    f"user index: {i}, team_id: {agent.metadata['team_id']}")
            else:
                self.teams.append(
                    AIcrowdAgentTeam(agent.metadata["team_id"],
                                     self.env_config,
                                     agent,
                                     policy_id=agent.metadata["policy_id"],
                                     act_timeout=100000))

    def finalize(self) -> None:
        super().finalize()

        for p in self.ps:
            p.join()

    @staticmethod
    def _find_user_policy_ids(policy_ids: List[str]) -> List[str]:
        return [x for x in policy_ids if not x.startswith("_bar_")]

    @staticmethod
    def check(shared_dir: str, mode: Mode):
        if mode == Mode.PVE_STAGE2:
            mode = Mode.PVE_STAGE1
        elif mode == Mode.PVE_STAGE3:
            mode = Mode.PVE_STAGE2
        elif mode == Mode.PVE_BONUS:
            mode = Mode.PVE_STAGE1

        history = CompetitionEvaluator.load(shared_dir, mode)
        if not history:
            return False

        topn_probs = analyzer.topn_prob_by_policy(history)
        logger.info(f"{topn_probs}")

        policy_ids = PVEEvaluator._find_user_policy_ids(list(
            topn_probs.keys()))
        assert len(policy_ids
                   ) == 1, f"multiple policy {policy_ids} exist in mode {mode}"
        policy_id = policy_ids[0]

        if topn_probs[policy_id] >= PVEEvaluator.pass_values[mode]:
            return True

        return False

    @staticmethod
    def generate_markdown(shared_dir: str,
                          modes: List[Mode],
                          replay_url: str = ""):
        text = markdown.result_head()
        for mode in modes:
            history: List[Dict[int, TeamResult]] = CompetitionEvaluator.load(
                shared_dir, mode)
            if not history:
                continue
            final_metrices = metrics.get_final_metrices(history, mode)

            n_timeout = analyzer.n_timeout(history)
            policy_ids = PVEEvaluator._find_user_policy_ids(
                list(n_timeout.keys()))
            assert len(
                policy_ids
            ) == 1, f"multiple policy {policy_ids} exist in mode {mode}"
            policy_id = policy_ids[0]

            pass_value = PVEEvaluator.pass_values.get(mode, 0.0)
            text += markdown.stage_result(mode, final_metrices,
                                          n_timeout[policy_id], pass_value)

        if replay_url:
            text += markdown.replay(replay_url)

        if text:
            util.write_data(
                text,
                os.path.join(shared_dir, "status.md"),
            )

    @staticmethod
    def evaluate(shared_dir: str, local: bool):
        modes = [
            Mode.PVE_STAGE1, Mode.PVE_STAGE2, Mode.PVE_STAGE3, Mode.PVE_BONUS
        ]

        ret = {"score": 0, "score_secondary": 0, "meta": {}}
        policy_id = None
        for mode in modes:
            history: List[Dict[int, TeamResult]] = CompetitionEvaluator.load(
                shared_dir, mode)
            if not history:
                continue
            final_metrices = metrics.get_final_metrices(history, mode)
            policy_ids = PVEEvaluator._find_user_policy_ids(
                list(final_metrices.keys()))
            assert len(policy_ids) == 1
            policy_id = policy_ids[0]
            ret["meta"].update({
                f"{mode}-{k}": v
                for k, v in final_metrices[policy_id].items()
            })

        replay = ""
        if not local:
            try:
                # cannot import on top of file, since it needs some env vars that are only
                # set in score
                from pkg import cos
                upload_ret = cos.upload_pve_replays(policy_id,
                                                    folder=os.path.join(
                                                        shared_dir, "replays"))
            except:
                logger.exception("upload replays failed")
            else:
                ret["meta"]["Replay"] = upload_ret.get("zip", "none")
                for mode in upload_ret.get("modes", {}):
                    ret["meta"][f"Replay-{mode}"] = upload_ret.get(
                        "modes", {}).get(f"{mode}", "none")

        util.write_data(
            json.dumps(ret, indent=2),
            os.path.join(shared_dir, "result.json"),
        )

        PVEEvaluator.generate_markdown(shared_dir, modes, replay)

        return ret
