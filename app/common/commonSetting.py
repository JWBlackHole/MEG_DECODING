from enum import Enum

class TargetLabel(Enum):
    """
    control the flow in meg signal preprocessing
    """
    DEFAULT = 0
    VOICED = 1
    IS_WORD = 2
    IS_SOUND = 3