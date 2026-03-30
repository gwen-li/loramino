import torch.nn as nn
from abc import ABC
from typing import Any


class Model(nn.Module):
    model: nn.Module
    tokenizer: Any

