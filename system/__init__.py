from typing import *

from .base import BaseSystem
from .tinynerf import TinyNerf

def parse_system(system_type:str) -> BaseSystem:
    if system_type == 'TinyNerf':
        return TinyNerf
    else:
        raise ValueError(f'[ERROR]: System {system_type} currently not supported')