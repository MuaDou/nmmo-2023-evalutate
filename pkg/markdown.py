from pkg.mode import Mode
from pkg.team import ACT_TIMEOUT

stage_result_template = r"""
### {stage}

{tip}

| Policy | Top1Ratio | TotalScore | AliveScore | DefeatScore | TimeAlive | Gold | DamageTaken |
|  ----  | ----      | ----       | ----       | ----        | ----      | ---- | ----        |
"""


def result_head():
    return r"""
## Result

    """


def stage_result(mode, metrices, n_timeout=0, pass_value=0.0):

    def get(m, k):
        r = m[k]
        if isinstance(r, float):
            return f"{r:.2f}"
        return r

    if mode == Mode.PVE_STAGE1:
        tip = f"> Need **Top1Ratio >= {pass_value}** to pass."
        stage = "PvE Stage1 :guardsman:"
    elif mode == Mode.PVE_STAGE2:
        tip = f"> Need **Top1Ratio >= {pass_value}** to pass."
        stage = "PvE Stage2 :space_invader:"
    elif mode == Mode.PVE_STAGE3:
        tip = ""
        pass_value = 0
        stage = "PvE Stage3 :alien:"
    elif mode == Mode.PVE_BONUS:
        tip = ""
        pass_value = 0
        stage = "PvE Bonus :video_game:"
    else:
        assert 0, f"invalid mode {mode}"
    if n_timeout:
        tip += "\n"
        tip += f"#### **Timeout(> {ACT_TIMEOUT}ms)** :snail: happened in {n_timeout} timesteps."
        tip += "\n\n"
        tip += "*The result below may not represent the actual skill of you policy :cry:. Please try making ``act()`` faster :sweat_drops:.*"
    r = "" + stage_result_template.format(stage=stage, tip=tip)

    for policy, m in metrices.items():
        if not policy.startswith("_bar_"):
            policy = "**``You``**"
            p = m["Top1Prob"]
            if pass_value > 0:
                if p < pass_value:
                    policy += " :sob:"
                else:
                    policy += " :sunglasses:"
            else:
                policy += " :fire:"
        else:
            policy = policy.replace('_bar_', '')
            policy = f"``{policy}``"
        r += f"| {policy} | {get(m, 'Top1Prob')} | {get(m, 'TotalScore')} | {get(m, 'AliveScore')} | {get(m, 'DefeatScore')} | {get(m, 'TimeAlive')} | {get(m, 'Gold')} | {get(m, 'DamageTaken')} |\n"

    r += "\n"
    return r


def replay(replay_url: str):
    return r"""
## Replay :movie_camera:

Download the replay zip file, click [here]({replay_url}).
    """.format(replay_url=replay_url)
