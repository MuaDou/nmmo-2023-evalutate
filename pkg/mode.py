from pkg import enum


class Mode(enum.Enum):
    PVE_STAGE1 = enum.auto()
    PVE_STAGE2 = enum.auto()
    PVE_STAGE3 = enum.auto()
    PVE_BONUS = enum.auto()

    PVE_STAGE1_VALIDATION = enum.auto()
    PVE_STAGE2_VALIDATION = enum.auto()
    PVE_STAGE3_VALIDATION = enum.auto()
    PVE_BONUS_VALIDATION = enum.auto()

    PVP = enum.auto()
    PVE_AGGREGATE = enum.auto()
    PVP_AGGREGATE = enum.auto()
