from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "google/pegasus-large"
    max_input_length: int = 512
    max_target_length: int = 128
    num_beams: int = 5
    early_stopping: bool = True
    cache_dir: Optional[str] = None
