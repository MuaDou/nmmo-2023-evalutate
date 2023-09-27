from collections import defaultdict
import os
import json
import nmmo
import shutil
import pickle
from loguru import logger
from typing import Dict, List
# from neurips2022nmmo import CompetitionConfig, RollOut, Team, TeamResult, timer, analyzer
# from aicrowd_gym.servers.zmq_oracle_server import ZmqOracleServer

# from pkg.serializer import PickleSerializer
from pkg.mode import Mode
from pkg import util
# from pkg.metrics import camelize
from pkg.evaluator.team_result import FinalTeamResult
from pkg import timer


# class Config(CompetitionConfig):
#     SAVE_REPLAY = "nmmo"
#     MAP_N = 400


class CompetitionEvaluator:
    host: str
    port: int
    _auto_ready: bool
    env_config: nmmo.config.Config
    # server: ZmqOracleServer
    user_team_indices: List[int]
    # teams: List[Team]
    # ro: RollOut

    def __init__(self,
                 host: str,
                 port: int,
                 _auto_ready: bool = False,
                 **kwargs) -> None:
        self.sig = util.random_string(9)
        self.host, self.port = host, port
        self._auto_ready = _auto_ready
        # self.env_config = Config()
        self.env_config.SAVE_REPLAY = f"replay-{self.sig}"
        # self.server = ZmqOracleServer(host, port, len(self.env_config.PLAYERS),
        #                               PickleSerializer())

        self._init_teams()
        # self.ro = RollOut(self.env_config,
        #                   self.teams,
        #                   parallel=True,
        #                   show_progress=False)

    def _init_teams(self):
        raise NotImplementedError

    # def run(self,
    #         max_steps_per_episode: int = 20,
    #         max_episodes: int = 1) -> List[Dict[int, TeamResult]]:
    #     with timer.count("rollout.run", printout=True):
    #         return self.ro.run(max_steps_per_episode, max_episodes)

    def finalize(self) -> None:
        logger.info("close agents")
        self.server.close_agents()
        logger.info("close agents done")
        
    @staticmethod
    def save_contenders(contenders, json_file_path):
        with open(json_file_path, 'w') as f:
            json.dump(contenders,f)
            print( "Contenser Json Dump Done" )

    @staticmethod
    def load_contenders(json_file_path ):
        with open( json_file_path ) as f:
            contenders = json.load(f)
            return contenders
    
    @staticmethod
    def load(shared_dir: str, rollout_name: str) -> List[Dict[int, FinalTeamResult]]:
        import glob
        filepaths = glob.glob(
            os.path.join(shared_dir, "results", f"result-{rollout_name}-*.pkl"))
        
        history: Dict[str, List[FinalTeamResult]] = defaultdict(list)
        for filepath in filepaths:
            with open(filepath, "rb") as fp:
                try:
                    data = fp.read()
                    ms = pickle.loads(data)
                except:
                    logger.exception(f"read {filepath} failed")
                else:
                    for policy_id, res_list in ms.items():
                        history[policy_id].extend( res_list )
        return history

    # def save(self, shared_dir: str, history: List[Dict[int, TeamResult]],
    #          mode: Mode, save_replay: bool):
    #     logger.info("start save")
    #     policy_ids = list(analyzer.topn_count_by_policy(history).keys())

    #     if save_replay:
    #         replay_path = f"{self.env_config.SAVE_REPLAY}.lzma"

    #         def overwrite_replay(replay_path, results: Dict[int, TeamResult]):
    #             metrics = {}
    #             for name in TeamResult.names():
    #                 metrics[camelize(name)] = {
    #                     i: getattr(r, name)
    #                     for i, r in results.items()
    #                 }

    #             import lzma
    #             with open(replay_path, "rb") as fp:
    #                 data = fp.read()
    #             data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
    #             replay = json.loads(data.decode("utf-8"))
    #             replay["metrics"] = metrics

    #             data = lzma.compress(json.dumps(replay).encode("utf-8"),
    #                                  format=lzma.FORMAT_ALONE)
    #             util.write_data(data, replay_path, True)

    #         overwrite_replay(replay_path, history[-1])

    #         replay_dir = os.path.join(shared_dir, "replays")
    #         if mode == Mode.PVP:
    #             path = os.path.join(
    #                 replay_dir,
    #                 f"replay-{mode}-{self.sig}-{'-'.join(policy_ids)}.lzma")
    #         else:
    #             path = os.path.join(replay_dir, f"replay-{mode}.lzma")
    #         os.system(f"mkdir -p {replay_dir}")
    #         shutil.move(replay_path, path)

    #     result_dir = os.path.join(shared_dir, "results")
    #     os.system(f"mkdir -p {result_dir}")
    #     result_path = os.path.join(result_dir, f"result-{mode}-{self.sig}.pkl")
    #     logger.info("dump result")
    #     util.write_data(
    #         pickle.dumps(history),
    #         result_path,
    #         binary=True,
    #     )
    #     logger.info("dump result done")
