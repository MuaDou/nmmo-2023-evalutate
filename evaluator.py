from collections import defaultdict
import os
import queue
import time
from typing import Dict, List
import dateutil
import pickle
import json
import tempfile
import threading
from loguru import logger

from pkg import (
    Mode,
    # PVEStage1Evaluator,
    # PVEStage2Evaluator,
    # PVEBonusEvaluator,
    PVPEvaluator,
    # PVEEvaluator,
    CompetitionEvaluator,
    util
)

from pkg.timer import timer

import argparse
from aicrowd_api import API, AIcrowdSubmission
import logging
import os, sys

import time
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from itertools import cycle

import numpy as np
import torch
import pandas as pd

from nmmo.render.replay_helper import FileReplayHelper

import pufferlib
from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import pufferlib.policy_ranker
import pufferlib.utils
import clean_pufferl

import environment

from reinforcement_learning import config
import subprocess

class Constants:
    LOCAL: bool = bool(int(os.getenv("NMMO_LOCAL", "0")))

    SERVER_HOST: str = os.getenv("AICROWD_REMOTE_SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("AICROWD_REMOTE_SERVER_PORT", "5000"))

    # shared directory mounted only in evaluation server across runs
    SHARED_DIR: str = os.getenv("AICROWD_SHARED_DIR", '/puffertank/share')

    # shared directory mounted in client and server across all runs
    PUBLIC_DIR: str = os.getenv("AICROWD_PUBLIC_SHARED_DIR",
                                tempfile.mkdtemp())

    MODE: Mode = Mode(os.getenv("c", 'PVP'))

    if MODE in [Mode.PVP, Mode.PVP_AGGREGATE]:
        GRADER_IDS = os.getenv("AICROWD_GRADER_IDS", "").split(",")
        API_TOKEN = os.getenv("AICROWD_API_TOKEN", "askdjslkj")
        NUM_PARALLEL_MATCH = int(os.getenv("NMMO_NUM_PARALLEL_MATCH", "1"))
        NUM_ROUNDS = int(os.getenv("NMMO_NUM_ROUNDS", "1"))
        MIN_SCORE = float(os.getenv("NMMO_MIN_SCORE", "0.01"))
        TOURNAMENT = os.getenv("NMMO_TOURNAMENT", "test")
        DEADLINE = int(os.getenv("NMMO_DEADLINE", 123456))
        UPLOAD_REPLAY = bool(int(os.getenv("NMMO_UPLOAD_REPLAY", "1")))
        DRY_RUN = bool(int(os.getenv("NMMO_PVP_AGGREGATE_DRY_RUN", "0")))
        SPECIFIC_SUBMISSIONS = [
            x.strip()
            for x in os.getenv("NMMO_SPECIFIC_SUBMISSIONS", "").split(",")
            if x.strip()
        ]
        MIN_REPLAY = int(os.getenv("NMMO_PVP_MIN_REPLAY", 30))
        # NMMO_BASE_POLICY_STORE_DIR: str = os.getenv("NMMO_POLICY_STORE_DIR", tempfile.mkdtemp())
        # NMMO_SUBMISSION_DIR: str = os.getenv("NMMO_SUBMISSION_DIR", tempfile.mkdtemp())
        # NMMO_ROLLOUT_NAME: str = os.getenv("NMMO_ROLLOUT_NAME", "TEMP")
        NMMO_BASE_POLICY_STORE_DIR: str = os.getenv("NMMO_POLICY_STORE_DIR", '/puffertank/policy_store_dir/')
        NMMO_SUBMISSION_DIR: str = os.getenv("NMMO_SUBMISSION_DIR", '/puffertank/submissions')
        NMMO_ROLLOUT_NAME: str = os.getenv("NMMO_ROLLOUT_NAME", "TEST")
        CONTENDER_JSON_PATH: str = os.getenv("CONTENDER_JSON_PATH", os.path.join( NMMO_ROLLOUT_NAME, 'Contender.json' ))
        TASK_FILE_PATH: str = os.getenv( "TASK_FILE_PATH", "/scripts/eval_task_with_embedding.pkl" ) 
        DOWNLOAD_SUBMISSION_SCRIPT_PATH: str = os.getenv( "DOWNLOAD_SUBMISSION_SCRIPT_PATH", "/scripts/setup.sh" )
        
        
        assert GRADER_IDS
        assert API_TOKEN
    # else:
    #     PVEEvaluator.pass_values[Mode.PVE_STAGE1] = float(
    #         os.getenv("NMMO_PVE_STAGE1_PASS_VALUE", 0.0))
    #     PVEEvaluator.pass_values[Mode.PVE_STAGE2] = float(
    #         os.getenv("NMMO_PVE_STAGE2_PASS_VALUE", 0.0))
    #     PVEEvaluator.pass_values[Mode.PVE_STAGE3] = float(
    #         os.getenv("NMMO_PVE_STAGE3_PASS_VALUE", 0.0))

    MAX_STEPS_PER_EPISODE = int(os.getenv("NMMO_MAX_STEPS_PER_EPISODE", 1024))

    SAVE_REPLAY = int(os.getenv("NMMO_SAVE_REPLAY", "0"))

class AllPolicySelector(pufferlib.policy_ranker.PolicySelector):
    def select_policies(self, policies):
        # Return all policy names in the alpahebetical order
        # Loops circularly if more policies are needed than available
        loop = cycle([
            policies[name] for name in sorted(policies.keys()
        )])
        return [next(loop) for _ in range(self._num)]

class AIcrowdEvaluator:
    host: str
    port: int
    evaluator: CompetitionEvaluator

    def __init__(self, **kwargs) -> None:
        if "host" in kwargs:
            self.host = kwargs["host"]
        else:
            self.host = Constants.SERVER_HOST

        if "port" in kwargs:
            self.port = int(kwargs["port"])
        else:
            self.port = int(Constants.SERVER_PORT)

    # def _init_evaluator(self):
        # if Constants.MODE == Mode.PVE_STAGE1:
        #     self.evaluator = PVEStage1Evaluator(self.host, self.port)
        # elif Constants.MODE == Mode.PVE_STAGE2:
        #     self.evaluator = PVEStage2Evaluator(self.host, self.port)
        # elif Constants.MODE == Mode.PVE_BONUS:
        #     self.evaluator = PVEBonusEvaluator(self.host, self.port)
        # elif Constants.MODE == Mode.PVP:
        # if Constants.MODE == Mode.PVP:
        #     self.evaluator = PVPEvaluator(self.host, self.port)
        # else:
        #     assert 0
          
    def setup_policy_store(self, policy_store_dir):
        # CHECK ME: can be custom models with different architectures loaded here?
        if not os.path.exists(policy_store_dir):
            raise ValueError("Policy store directory does not exist")
        if os.path.exists(os.path.join(policy_store_dir, "trainer.pt")):
            raise ValueError("Policy store directory should not contain trainer.pt")
        logging.info("Using policy store from %s", policy_store_dir)
        policy_store = DirectoryPolicyStore(policy_store_dir)
        return policy_store

    def save_replays(self, policy_store_dir, save_replay_dir):
        # load the checkpoints into the policy store
        policy_store = self.setup_policy_store(policy_store_dir)
        num_policies = len(policy_store._all_policies())

        # setup the replay path
        save_dir = save_replay_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.info("Replays will be saved to %s", save_dir)

        # Use 1 env and 1 buffer for replay generation
        # TODO: task-condition agents when generating replays
        args = SimpleNamespace(**config.Config.asdict())
        args.num_envs = 1
        args.num_buffers = 1
        args.use_serial_vecenv = True
        args.learner_weight = 0  # evaluate mode
        args.selfplay_num_policies = num_policies + 1
        args.early_stop_agent_num = 0  # run the full episode
        args.resilient_population = 0  # no resilient agents

        # NOTE: This creates a dummy learner agent. Is it necessary?
        from reinforcement_learning import policy  # import your policy
        def make_policy(envs):
            learner_policy = policy.Baseline(
                envs.driver_env,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                task_size=args.task_size
            )
            return cleanrl.Policy(learner_policy)

        # Setup the evaluator. No training during evaluation
        evaluator = clean_pufferl.CleanPuffeRL(
            seed=args.seed,
            env_creator=environment.make_env_creator(args),
            env_creator_kwargs={},
            agent_creator=make_policy,
            vectorization=Serial,
            num_envs=args.num_envs,
            num_cores=args.num_envs,
            num_buffers=args.num_buffers,
            selfplay_learner_weight=args.learner_weight,
            selfplay_num_policies=args.selfplay_num_policies,
            policy_store=policy_store,
            data_dir=save_dir,
        )

        # Load the policies into the policy pool
        evaluator.policy_pool.update_policies({
            p.name: p.policy(
                policy_args=[evaluator.buffers[0]], 
                device=evaluator.device
            ) for p in list(policy_store._all_policies().values())
        })

        # Set up the replay helper
        o, r, d, i = evaluator.buffers[0].recv()  # reset the env
        replay_helper = FileReplayHelper()
        nmmo_env = evaluator.buffers[0].envs[0].envs[0].env
        nmmo_env.realm.record_replay(replay_helper)
        replay_helper.reset()

        # Run an episode to generate the replay
        while True:
            with torch.no_grad():
                actions, logprob, value, _ = evaluator.policy_pool.forwards(
                    torch.Tensor(o).to(evaluator.device),
                    None,  # dummy lstm state
                    torch.Tensor(d).to(evaluator.device),
                )
                value = value.flatten()
            evaluator.buffers[0].send(actions.cpu().numpy(), None)
            o, r, d, i = evaluator.buffers[0].recv()

            num_alive = len(nmmo_env.realm.players)
            print('Tick:', nmmo_env.realm.tick, ", alive agents:", num_alive)
            if num_alive == 0 or nmmo_env.realm.tick == args.max_episode_length:
                break

        # Save the replay file
        replay_file = os.path.join(save_dir, f"replay-{Constants.NMMO_ROLLOUT_NAME}-{time.strftime('%Y%m%d_%H%M%S')}")
        logging.info("Saving replay to %s", replay_file)
        replay_helper.save(replay_file, compress=True)
        evaluator.close()

    def create_policy_ranker(self, policy_store_dir, ranker_file="openskill.pickle"):
        file = os.path.join(policy_store_dir, ranker_file)
        if os.path.exists(file):
            if os.path.exists(file + ".lock"):
                raise ValueError("Policy ranker file is locked. Delete the lock file.")
            logging.info("Using policy ranker from %s", file)
            policy_ranker = pufferlib.utils.PersistentObject(
                file,
                pufferlib.policy_ranker.OpenSkillRanker,
            )
        else:
            policy_ranker = pufferlib.utils.PersistentObject(
                file,
                pufferlib.policy_ranker.OpenSkillRanker,
                "anchor",
            )
        return policy_ranker

    def rollout(self, policy_store_dir, eval_curriculum_file, save_result_dir, device):
        # CHECK ME: can be custom models with different architectures loaded here?
        policy_store = self.setup_policy_store(policy_store_dir)
        policy_ranker = self.create_policy_ranker(policy_store_dir)
        num_policies = len(policy_store._all_policies())
        policy_selector = AllPolicySelector(num_policies)

        args = SimpleNamespace(**config.Config.asdict())
        args.data_dir = policy_store_dir
        args.eval_mode = True
        args.num_envs = 5  # sample a bit longer in each env
        args.num_buffers = 1
        args.learner_weight = 0  # evaluate mode
        args.selfplay_num_policies = num_policies + 1
        args.early_stop_agent_num = 0  # run the full episode
        args.resilient_population = 0  # no resilient agents
        args.tasks_path = eval_curriculum_file# task-conditioning

        # TODO: custom models will require different policy creation functions
        from reinforcement_learning import policy  # import your policy
        def make_policy(envs):
            learner_policy = policy.Baseline(
                envs.driver_env,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                task_size=args.task_size
            )
            return cleanrl.Policy(learner_policy)
         # Setup the evaluator. No training during evaluation
        evaluator = clean_pufferl.CleanPuffeRL(
            device=torch.device(device),
            seed=args.seed,
            env_creator=environment.make_env_creator(args),
            env_creator_kwargs={},
            agent_creator=make_policy,
            data_dir=policy_store_dir,
            vectorization=Multiprocessing,
            num_envs=args.num_envs,
            num_cores=args.num_envs,
            num_buffers=args.num_buffers,
            selfplay_learner_weight=args.learner_weight,
            selfplay_num_policies=args.selfplay_num_policies,
            batch_size=args.eval_batch_size,
            policy_store=policy_store,
            policy_ranker=policy_ranker, # so that a new ranker is created
            policy_selector=policy_selector,
        )

        rank_file = os.path.join(policy_store_dir, "ranking.txt")
        with open(rank_file, "w") as f:
            pass

        results = defaultdict(list)
        # while evaluator.global_step < args.eval_num_steps:
        while evaluator.global_step < 1:
            _, stats, infos = evaluator.evaluate()

            for pol, vals in infos.items():
                results[pol].extend([
                    e[1] for e in infos[pol]['team_results']
                ])

            ratings = evaluator.policy_ranker.ratings()
            dataframe = pd.DataFrame(
                {
                    ("Rating"): [ratings.get(n).mu for n in ratings],
                    ("Policy"): ratings.keys(),
                }
            )

            with open(rank_file, "a") as f:
                f.write(
                    "\n\n"
                    + dataframe.round(2)
                    .sort_values(by=["Rating"], ascending=False)
                    .to_string(index=False)
                    + "\n\n"
                )

            # Reset the envs and start the new episodes
            # NOTE: The below line will probably end the episode in the middle, 
            #   so we won't be able to sample scores from the successful agents.
            #   Thus, the scores will be biased towards the agents that die early.
            #   Still, the numbers we get this way is better than frequently
            #   updating the scores because the openskill ranking only takes the mean.
            #evaluator.buffers[0]._async_reset()

            # CHECK ME: delete the policy_ranker lock file
            Path(evaluator.policy_ranker.lock.lock_file).unlink(missing_ok=True)

        evaluator.close()
        
        # print( results )
        
        # Save the result file
        result_dir = save_result_dir
        print( result_dir )
        os.system(f"mkdir -p {result_dir}")
        result_path = os.path.join(result_dir, f"result-{Constants.NMMO_ROLLOUT_NAME}-{time.strftime('%Y%m%d_%H%M%S')+'.pkl'}")
        print( result_path )
        logger.info("dump result")
        util.write_data(
            pickle.dumps(results),
            result_path,
            binary=True,
        )
        logger.info("dump result done")
        
        return results

    def serve(self, **kwargs):
        with timer.count(f"serve", printout=True):
            logger.info(f"Running in {Constants.MODE} mode")
        
            # TODO: Pass in the task embedding?
            logging.basicConfig(level=logging.INFO)
            
            policy_store_dir = Constants.NMMO_BASE_POLICY_STORE_DIR
            assert policy_store_dir is not None, "Policy store directory must be specified"
            replay_save_dir = os.path.join( Constants.SHARED_DIR, 'replays' )
            save_result_dir = os.path.join( Constants.SHARED_DIR, 'results' )
            device = 'cpu'
            task_file = Constants.TASK_FILE_PATH
            
            assert policy_store_dir is not None, "Policy store directory must be specified"
            with timer.count(f"evaluator.run", printout=True):
                # TODO: rollout gen result
                # history = self.evaluator.run(Constants.MAX_STEPS_PER_EPISODE)、
                logging.info("Ranking checkpoints from %s", policy_store_dir)
                res = self.rollout( policy_store_dir, task_file, save_result_dir, device )
                logging.info("Replays will NOT be generated")
                
            
            # TODO: gen replays
            if Constants.MODE == Mode.PVP:
                self.save_replays(policy_store_dir, replay_save_dir)
            # else:
            #     self.evaluator.save(Constants.SHARED_DIR, history,
            #                         Constants.MODE, Constants.SAVE_REPLAY)
            # threading.Thread(target=self.evaluator.finalize,
            #                  daemon=True).start()
            # time.sleep(3)

        logger.info(f"Time cost stat:\n{json.dumps(timer.reset(), indent=2)}")
        Constants.MODE = Mode.PVP_AGGREGATE

    # def evaluate(self, **kwargs):
    #     if Constants.MODE != Mode.PVE_AGGREGATE:
    #         logger.error(f"calling evaluate in {Constants.MODE} mode")
    #         return {}

    #     ret = PVEEvaluator.evaluate(Constants.SHARED_DIR, Constants.LOCAL)

    #     if not Constants.LOCAL:
    #         self.wait_for_gitlab_sync()

    #     return ret

    def aggregate_results(self, *args, **kwargs):
        if Constants.MODE != Mode.PVP_AGGREGATE:
            logger.error(f"calling evaluate in {Constants.MODE} mode")
            return {}

        ret = PVPEvaluator.evaluate(Constants.SHARED_DIR, Constants.NMMO_ROLLOUT_NAME, Constants.API_TOKEN, Constants.LOCAL, Constants.TOURNAMENT, Constants.CONTENDER_JSON_PATH,Constants.UPLOAD_REPLAY, Constants.DRY_RUN)

        return ret

    @staticmethod
    def schedule_matches(self, *args, **kwargs):
        # remove all the policy in policy_store_dir
        # self.reset_policy_store()
        
        q = queue.Queue()
        submission_q = queue.Queue()

        def get_submissions(grader_id: str, thread_id: int):
            api = API(Constants.API_TOKEN)
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

        api = API(Constants.API_TOKEN)
        for grader_id in Constants.GRADER_IDS:
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

        for submissions in participant_submissions.values():
            latest = None
            chosen = None
            if Constants.SPECIFIC_SUBMISSIONS:
                for subm in submissions:
                    if str(subm["id"]) in Constants.SPECIFIC_SUBMISSIONS:
                        chosen = subm
                        break
            else:
                for subm in submissions:
                    ts = dateutil.parser.parse(subm["created_at"]).timestamp()
                    if ts > Constants.DEADLINE:
                        continue
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
                        if ts > Constants.DEADLINE:
                            continue
                        if latest is None or (latest is not None
                                              and ts > latest):
                            latest = ts
                            chosen = subm
                else:
                    logger.info(
                        f"choose specific submission for pvp: {chosen}")

            if chosen:
                # cp to policy_store
                self.copy_to_policy_store( chosen['participant_id'] )
        
        # cp the submission to policy_dir
        # matches = PVPEvaluator.schedule_matches(
        #     Constants.SHARED_DIR, Constants.GRADER_IDS, Constants.API_TOKEN,
        #     Constants.NUM_PARALLEL_MATCH, Constants.MIN_SCORE,
        #     Constants.NUM_ROUNDS, Constants.DEADLINE,
        #     Constants.SPECIFIC_SUBMISSIONS)
        # return matches
        
    def reset_policy_store(self):
        os.system( f"cd {Constants.NMMO_BASE_POLICY_STORE_DIR}" )
        os.system( f"rm -rf *" )
        
    def copy_to_policy_store(self, submission_id):
        # 移动.pt文件而不是id
        os.system( f"cp {submission_id} {Constants.NMMO_BASE_POLICY_STORE_DIR}" )

    @staticmethod
    def wait_for_gitlab_sync():
        """
        If the evaluator finished within 1 min, this will force the class to
        not quit before 1 min. This 1 min hold is needed as a hack to let the
        final sync on gitlab finish.
        """
        time.sleep(60)

    @staticmethod
    def gen_contenders():
        from aicrowd_api import API

        api_token = os.getenv( "AICROWD_API_TOKEN" )
        api = API(api_token)
        # api.authenticate_participant("b21e87935a9a0053f2f727b023e18d64")
        grader_id = os.getenv('AICROWD_GRADER_IDS')
            
        submission_ids = api.get_all_submissions(grader_id, grading_status='*')
        submission_ids.sort(reverse=True)

        infos = defaultdict(lambda x: defaultdict)
        for submisson_id in submission_ids:
            submisson = api.get_submission(grader_id, submisson_id).raw_response
            if submisson:
                # print(submisson)
                
                if submisson['participant_id'] in infos:
                    continue
                
                if submisson['meta'] is not None and submisson['meta']['repo_url'] not in infos.values():
                    if not submisson['participant_id']:
                        continue
                    infos[submisson['participant_id']] = {
                        'repo_url': 'git@' + submisson['meta']['repo_url'].split('http://')[-1].replace( 'com/', 'com:' ),
                        'participant_name': submisson['participant_name'],
                        'tag': submisson['meta']['repo_ref'].split( 'tags/' )[-1],
                        'name': submisson['participant_name'],
                        'id': submisson['id'],
                        'grader_id': grader_id,
                        'submit_id': submisson['id']
                    }
        return infos

    @staticmethod         
    def folder_check( policy_store_dir, submissions_dir, shared_dir ):
        if os.path.exists( policy_store_dir ):
            print( 'Remove Previous Policy Store Dir' )
            # os.rmdir( policy_store_dir )
            os.system( f'rm -rf {policy_store_dir}' )
        print( 'Creat Policy Store Dir' )
        os.makedirs( policy_store_dir )
        
        if os.path.exists( submissions_dir ):
            print( 'Remove Previous Submissions Dir' )
            # os.rmdir( submissions_dir )
            os.system( f'rm -rf {submissions_dir}' )
        print( 'Creat Submissions Dir' )
        os.makedirs( submissions_dir )
        
        if os.path.exists( shared_dir ):
            print( 'Remove Previous Share Dir' )
            # os.rmdir( shared_dir )
            os.system( f'rm -rf {shared_dir}' )
        print( 'Creat Share Dir' )
        os.makedirs( shared_dir )
    
    @staticmethod
    def setup_pufferlib():
        contenders = CompetitionEvaluator.load_contenders( Constants.CONTENDER_JSON_PATH )
        
        if not os.path.exists( Constants.TASK_FILE_PATH ):
            print( "bash Task File not exist." )
            sys.exit(1)
        
        if not os.path.exists( Constants.DOWNLOAD_SUBMISSION_SCRIPT_PATH ):
            print( "bash Download File not exist." )
            sys.exit(1)
        
        for key, info in contenders.items():
            url = info['repo_url']
            tag = info['tag']
            name = info['name']
            print( f"clone repo of {name}" )
            code = subprocess.call( f'bash {Constants.DOWNLOAD_SUBMISSION_SCRIPT_PATH} {tag} {url} {key} {Constants.NMMO_BASE_POLICY_STORE_DIR} {name} {Constants.NMMO_SUBMISSION_DIR}', shell=True )
        
            if code != 0:
                print("bash setup.sh failed.")
                sys.exit(1)