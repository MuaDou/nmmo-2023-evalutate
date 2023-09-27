import os
import multiprocessing
from loguru import logger
from typing import List
from neurips2022nmmo import Team, submission
from aicrowd_gym.clients.zmq_oracle_client import ZmqOracleClient

from pkg.serializer import PickleSerializer
from pkg.evaluator.pve import PVEEvaluator


class PVEBonusEvaluator(PVEEvaluator):
    keeper_path: str = "./keepers/bonus"

    def _start_baseline_teams(self) -> List[multiprocessing.Process]:
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        ret = os.system(
            f"tar -xf {self.keeper_path}-mkk.tar -C {os.path.dirname(self.keeper_path)}"
        )
        if ret:
            raise f"untar keeper policy failed"

        ret = os.system(
            f"tar -xf {self.keeper_path}-zz.tar -C {os.path.dirname(self.keeper_path)}"
        )
        if ret:
            raise f"untar keeper policy failed"

        ret = os.system(
            f"tar -xf {self.keeper_path}-here.tar -C {os.path.dirname(self.keeper_path)}"
        )
        if ret:
            raise f"untar keeper policy failed"

        team1 = submission.get_team_from_submission(
            f"{self.keeper_path}-mkk",
            "GuaGua(瓜瓜)",
            self.env_config,
        )
        team2 = submission.get_team_from_submission(
            f"{self.keeper_path}-zz",
            "Universe(宇宙)",
            self.env_config,
        )
        team3 = submission.get_team_from_submission(
            f"{self.keeper_path}-here",
            "Blunt(无锋)",
            self.env_config,
        )

        ps = []
        ps.extend(self._start_teams([team1] * 5))
        ps.extend(self._start_teams([team2] * 5))
        ps.extend(self._start_teams([team3] * 5))
        return ps

    def _start_teams(self, teams: List[Team]) -> List[multiprocessing.Process]:

        def _start(team: Team, i: int):
            team.policy_id = f"_bar_{team.id}"
            team.id = f"{team.id.split('(')[0]}-{i}"

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
        for i, team in enumerate(teams):
            p = multiprocessing.Process(target=_start,
                                        args=(team, i),
                                        daemon=True)
            p.start()
            ps.append(p)

        return ps
