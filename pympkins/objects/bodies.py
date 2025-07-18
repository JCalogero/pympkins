"""
This file contains the basic body classes.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ..constants import ORIGIN
from ..utils import NameMixin


@dataclass
class Frame(NameMixin):
    """A reference frame in space.
    """
    name: str
    fixed: bool = False

inertial_frame = Frame(name="inertial_frame", fixed=True)

@dataclass
class Point(NameMixin):
    """
    A vector from a reference frame to a point of interest.

    Args:
        name
            The name of the point.
        frame
            The reference frame in which the point is defined.
        position
            The position of the point in 3D space, represented as a numpy array.
        
    """
    name: str
    frame: Frame = field(default_factory=lambda: inertial_frame)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Body(NameMixin):
    """
    A solid object in space.

    Upon creation, a body will have an origin point defined in its own frame.

    Args:
        name
            The name of the body.
        mass
            The mass of the body.
        mmoi
            The moment of inertia matrix of the body, represented as a 3x3 numpy array.
    """
    # Public attributes
    name: str
    mass: float
    mmoi: np.ndarray

    # Property attributes
    _mass: float = field(init=False, repr=False)
    _mmoi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.origin = Point(name=ORIGIN, frame=Frame(name=self.name))
        
    # ----------
    # Properties
    # ----------
    @property
    def mass(self) -> float:
        return self._mass
    @mass.setter
    def mass(self, value: float):
        if value <= 0:
            raise ValueError("Mass must be a positive value.")
        self._mass = value

    @property
    def mmoi(self) -> np.ndarray:
        return self._mmoi
    @mmoi.setter
    def mmoi(self, value: np.ndarray):
        if value.shape != (3, 3):
            raise ValueError("Moment of inertia matrix must be a 3x3 numpy array.")
        self._mmoi = value

    @property
    def origin(self) -> Point:
        return self._origin
    @origin.setter
    def origin(self, value: Point):
        if not isinstance(value, Point):
            raise TypeError("Origin must be an instance of Point.")
        if value.frame.name != self.name:
            raise ValueError(f"Origin point must be in the '{self.name}' frame.")
        self._origin = value
