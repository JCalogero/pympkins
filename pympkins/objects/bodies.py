
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class Body:
    """
    Represents a body in the pympkins framework.

    Args:
        name
            The name of the body.
        mass
            The mass of the body.
        mmoi
            The moment of inertia matrix of the body, represented as a 3x3 numpy array.
        
    """
    name: str
    mass: float
    mmoi: np.ndarray
