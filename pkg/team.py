import nmmo
from typing import Dict
from neurips2022nmmo import Team, exception
from aicrowd_gym.servers.zmq_agent_server import ZmqAgent

TIMEOUT_RETURN = "TimeoutError"
ACT_TIMEOUT = 600
RESET_TIMEOUT = 30000


class AIcrowdAgentTeam(Team):
    agent: ZmqAgent

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        aicrowd_agent: ZmqAgent,
        reset_timeout=RESET_TIMEOUT,
        act_timeout=ACT_TIMEOUT,
        **kwargs,
    ) -> None:
        super().__init__(team_id, env_config, **kwargs)

        self.agent = aicrowd_agent
        self.reset_timeout = reset_timeout
        self.act_timeout = act_timeout
        self.timeout_act_req_id = ""

    def reset(self) -> None:
        req_id = self.agent.execute("reset")
        ret = self.agent.get_response(req_id,
                                      TIMEOUT_RETURN,
                                      timeout=self.reset_timeout)
        if ret == TIMEOUT_RETURN:
            raise exception.TeamTimeoutError(
                f"team[{self.id}] reset exceeds {self.reset_timeout}ms")

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        if self.timeout_act_req_id:
            ret = self.agent.get_response(self.timeout_act_req_id,
                                          TIMEOUT_RETURN,
                                          timeout=1)
            if ret == TIMEOUT_RETURN:
                return {}
            else:
                self.timeout_act_req_id = ""

        req_id = self.agent.execute("act", observations)
        ret = self.agent.get_response(req_id,
                                      TIMEOUT_RETURN,
                                      timeout=self.act_timeout)
        if ret == TIMEOUT_RETURN:
            self.timeout_act_req_id = req_id
            raise exception.TeamTimeoutError(
                f"team[{self.id}] act exceeds {self.act_timeout}ms")

        return ret
