from dataclasses import dataclass

@dataclass
class Trainconfig:
     train_frames: int = 4000
     val_frames: int = 600

     epochs: int = 20
     batch_size: int = 512

     learning_rate: float =  1e-3
     weight_decay: float = 1e-4

     hidden_dim: int = 128

     ebn0_min: float = 0.0
     ebn0_max: float = 20.0

     seed: int = 2026
     mode: str = "normal"