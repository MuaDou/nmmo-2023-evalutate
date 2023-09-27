import enum
from enum import auto  # noqa


class Enum(str, enum.Enum):

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._member_names_

    def _generate_next_value_(name, start, count, last_values):
        return name
