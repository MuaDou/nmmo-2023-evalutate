import os
import subprocess
import glob
import random
from loguru import logger
from typing import List
# from qcloud_cos import CosConfig
# from qcloud_cos import CosS3Client

# ENDPOINT = os.environ["COS_ENDPOINT"]
# GLOBAL_ENDPOINT = os.environ["COS_GLOBAL_ENDPOINT"]

# SECRET_KEY = os.environ["COS_SECRET_KEY"]
# SECRET_ID = os.environ["COS_SECRET_ID"]

# REGION = os.environ["COS_REGION"]
# BUCKET = os.environ["COS_BUCKET"]


# def upload(key: str, filepath: str) -> str:
#     config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
#     client = CosS3Client(config)

#     with open(filepath, "rb") as fp:
#         client.put_object(Bucket=BUCKET,
#                           Body=fp,
#                           Key=key,
#                           StorageClass='STANDARD',
#                           EnableMD5=False)

#     return f"{GLOBAL_ENDPOINT}/{key}"


def upload_pve_replays(submission_id: str, folder: str) -> dict:
    ret = {"modes": {}}
    replays = glob.glob(f"{folder}/replay-*.lzma")

    if not replays:
        logger.warning(f"no replays, folder: {folder}")
        return {}

    prefix = f"replays/submissions/{submission_id}"
    zipname = f"{folder}/replays-PVE-{submission_id}.zip"
    os.system(f"rm -f {zipname}; zip -j {zipname} {' '.join(replays)}")
    logger.info("upload zip")
    url = upload(f"{prefix}/{os.path.basename(zipname)}", zipname)
    ret["zip"] = url

    for replay in replays:
        mode = os.path.basename(replay).replace("replay-",
                                                "").replace(".lzma", "")
        url = upload(f"{prefix}/{os.path.basename(replay)}", replay)
        ret["modes"][mode] = url

    return ret


def upload_pvp_replays(tournament: str, folder: str,
                       submission_ids: List[str]) -> dict:
    ret = {"submissions": {}}
    replays = glob.glob(f"{folder}/replay-*.lzma")

    if not replays:
        logger.warning(f"no replays, folder: {folder}")
        return {}

    for submission_id in submission_ids:
        random.shuffle(replays)
        for replay in replays:
            if submission_id in replay:
                logger.info(
                    f"upload pvp {tournament} replay for {submission_id}")
                # url = upload(
                #     f"replays/submissions/{submission_id}/replay-PVP-{tournament}.lzma",
                #     replay)
                # ret["submissions"][f"{submission_id}"] = url
                break

    zipname = f"replays-PVP-{tournament}.zip"
    # use subprocess.run to capture error log
    # use pipe to avoid "Argument list too long"   
    subprocess.run(f"rm -f {zipname}; find {folder} -name 'replay-*.lzma' | zip {zipname} -@", shell=True, capture_output=True, check=True)
    # os.system(f"rm -f {zipname}; zip -j {zipname} {' '.join(replays)}")
    logger.info("upload zip")
    # url = upload(f"replays/pvp/{os.path.basename(zipname)}", zipname)
    # ret["zip"] = url

    return ret
