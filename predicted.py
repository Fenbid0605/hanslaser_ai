import dataclasses


@dataclasses.dataclass
class Predicted:
    current: int = 0
    speed: int = 0
    frequency: int = 0
    release: int = 0
    L: float = 0
    A: float = 0
    B: float = 0
    loss: float = 0
