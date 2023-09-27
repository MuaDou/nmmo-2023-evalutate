import time
import tempfile
import shutil
import string
import random
from loguru import logger

# from nmmo import Replay


def write_data(data: str, dst: str, binary: bool = False):
    mode = "wb" if binary else "w"
    with tempfile.NamedTemporaryFile(mode=mode, delete=False) as f:
        f.write(data)
    shutil.move(f.name, dst)


def random_string(size=14):
    return ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(size))


def guarantee(func, retry_interval: float = 1.0):

    def wrapper(*args, **kwargs):
        while 1:
            try:
                ret = func(*args, **kwargs)
            except:
                logger.exception(
                    f"calling {func} with args:{args} kwargs:{kwargs} failed")
                time.sleep(retry_interval)
            else:
                return ret

    return wrapper


# def submission_id_to_name(replay: Replay, mapping: dict) -> Replay:
#     for packet in replay.packets:
#         if "player" not in packet:
#             continue
#         for p in packet["player"].values():
#             if "base" in p and "name" in p["base"]:
#                 name: str = p["base"]["name"]
#                 subm_id, eid = name.split("_")
#                 p["base"]["name"] = mapping.get(subm_id, subm_id) + "_" + eid

#     return replay

# TODO: downlaod replay file to have a try
# def submission_id_to_name(replay: Replay, mapping: dict) -> Replay:
#     subm2eid = {} if len(mapping) < 16 else None
#     for packet in replay.packets:
#         if "player" not in packet:
#             continue
#         for key in sorted(packet["player"].keys()):
#             p = packet["player"][key]
#             if "base" in p and "name" in p["base"]:
#                 name: str = p["base"]["name"]
#                 subm_id, eid = name.split("_")
#                 if subm2eid is not None:
#                     if subm_id not in subm2eid:
#                         subm2eid[subm_id] = int(eid)
#                     if abs(int(eid) - subm2eid[subm_id]) < 8:
#                         p["base"]["name"] = mapping.get(subm_id,
#                                                         subm_id) + "-1_" + eid
#                     else:
#                         p["base"]["name"] = mapping.get(subm_id,
#                                                         subm_id) + "-2_" + eid
#                 else:
#                     p["base"]["name"] = mapping.get(subm_id, subm_id) + "_" + eid
#     return replay
