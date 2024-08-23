from enum import Enum

class TargetLabel(Enum):
    """
    control the flow in meg signal preprocessing
    """
    DEFAULT = 0
    VOICED_PHONEME = 1
    WORD_FREQ = 2
    IS_SOUND = 3