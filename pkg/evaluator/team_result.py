from typing import List

class FinalTeamResult:
    policy_id: str = None

    # event-log based, coming from process_event_log
    total_score: int = 0
    agent_kill_count: int = 0,
    npc_kill_count: int = 0,
    max_progress_to_center: int = 0,
    eat_food_count: int = 0,
    drink_water_count: int = 0,
    item_buy_count: int = 0,

    # agent object based (fill these in the environment)
    # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
    time_alive: int = 0,
    earned_gold: int = 0,
    completed_task_count: int = 0,
    damage_received: int = 0,
    ration_consumed: int = 0,
    potion_consumed: int = 0,
    
    @classmethod
    def names(cls) -> List[str]:
        return [
            "total_score",
            "agent_kill_count",
            "npc_kill_count",
            "max_progress_to_center",
            "eat_food_count",
            "drink_water_count",
            "item_buy_count",
            "time_alive",
            "earned_gold",
            "completed_task_count",
            "damage_received",
            "ration_consumed",
            "potion_consumed",
        ]
