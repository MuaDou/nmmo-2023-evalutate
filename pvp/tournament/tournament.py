import time
import subprocess
from loguru import logger

NFS_POD = "nfs-5fb5dbdc57-rhd8k"
TOURNAMENT = "final-top16"
DEFAULT_ARGS = "-n default --context nmmo"


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True)
    return r.returncode, r.stdout.decode().strip(), r.stderr.decode().strip()


def create_wf():
    code, out, err = run(
        f"helm template . | kubectl create -f - {DEFAULT_ARGS}")
    if code:
        raise Exception(f"create_wf failed.\nOut: {out}\nErr: {err}")

    wf = out.split()[0].split("/")[-1]
    logger.info(f"wf {wf} created")
    return wf


def check_wf(wf):
    code, out, err = run(f"kubectl get wf {wf} {DEFAULT_ARGS}")
    if code:
        raise Exception(f"check_wf failed.\nOut: {out}\nErr: {err}")

    if "Running" in out:
        return True
    return False


def delete_wf(wf):
    run(f"kubectl delete wf {wf} {DEFAULT_ARGS}")


# def get_finished_match_num():
#     code, out, err = run(
#         f"kubectl exec {NFS_POD} {DEFAULT_ARGS} -- bash -c 'ls /nfs/pvp/{TOURNAMENT}/results | wc -l'"
#     )
#     if code:
#         logger.warning(
#             f"get_finished_match_num failed.\nOut: {out}\nErr: {err}")
#         num = 0
#     else:
#         num = int(out)
#     logger.info(f"Number of finished matches: {num}")
#     return num


# get_finished_match_num()
wf = create_wf()
start = time.time()
while 1:
    time.sleep(30)
    try:
        running = check_wf(wf)
    except Exception as e:
        logger.info(e)
        running = False

    if running:
        logger.info(f"wf {wf} is running ...")
        continue
    # else:
    #     logger.info(f"exit!")
    #     break

    if time.time() - start > 300:
        delete_wf(wf)
        # get_finished_match_num()
        wf = create_wf()
        start = time.time()
    else:
        logger.info(f"wait till 300 seconds ...")
