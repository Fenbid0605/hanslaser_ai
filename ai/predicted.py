import dataclasses


@dataclasses.dataclass
class Predicted:
    frequency: int = 0
    speed: int = 0
    current: int = 0
    gap: int = 0
    L: float = 0
    A: float = 0
    B: float = 0
    loss: float = 0
