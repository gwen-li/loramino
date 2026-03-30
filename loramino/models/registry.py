from .base import Model
from .pythia import Pythia
from typing import Callable


model_dict: dict[str, Callable[[], Model]] = {
    'pythia-14m': lambda: Pythia(num_params='14m'),
    'pythia-31m': lambda: Pythia(num_params='31m'),
    'pythia-70m': lambda: Pythia(num_params='70m'),
    'pythia-160m': lambda: Pythia(num_params='160m'),
    'pythia-410m': lambda: Pythia(num_params='410m'),
    'pythia-1b': lambda: Pythia(num_params='1b'),
}

