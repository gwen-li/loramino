from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConfigOptions:
    model_name: str
    dataset: Any
    num_adapters: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    lora_config: Dict[str, Any]
    optimizer: str
    optimizer_params: Dict[str, Any]
    loss_function: str
    device: Optional[str] = None
    verbose: bool = False
