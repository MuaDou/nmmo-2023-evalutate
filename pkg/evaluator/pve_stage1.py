import multiprocessing
from typing import List

from neurips2022nmmo import scripted
from pkg.evaluator.pve import PVEEvaluator


class PVEStage1Evaluator(PVEEvaluator):

    def _start_baseline_teams(self) -> List[multiprocessing.Process]:
        teams = []
        teams.extend([
            scripted.CombatTeam(f"Combat-{i}",
                                self.env_config,
                                policy_id="_bar_Combat") for i in range(5)
        ])
        teams.extend([
            scripted.MixtureTeam(f"Mixture-{i}",
                                 self.env_config,
                                 policy_id="_bar_Mixture") for i in range(10)
        ])

        # for debug
        if self._auto_ready:
            teams.append(scripted.MeleeTeam(f"Melee", self.env_config))

        return self._start_teams(teams)
