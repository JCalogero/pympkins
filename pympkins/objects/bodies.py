"""
This file contains the basic body classes.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ..constants import ORIGIN


@dataclass
class Point:
    """
    A point in space.

    Args:
        name
            The name of the point.
        position
            The position of the point in 3D space, represented as a numpy array.
        
    """
    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


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
        origin
            The origin point of the body, represented as a Point object.
    """
    name: str
    mass: float
    mmoi: np.ndarray
    origin: Point = field(default_factory=lambda: Point(name=ORIGIN))
