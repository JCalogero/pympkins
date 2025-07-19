"""
This file contains the basic body classes.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..constants import ORIGIN
from ..utils import NameMixin


@dataclass
class Frame(NameMixin):
    """A reference frame in space."""

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
    _points: list[Point] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Create a frame for the body and an origin point
        self._frame = Frame(name=self.name, fixed=False)
        self._origin = Point(name=ORIGIN, frame=self.frame)

    # ----------
    # Properties
    # ----------
    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float):
        if value <= 0 or not np.isfinite(value) or np.isnan(value):
            raise ValueError("Mass must be a positive, non-NaN, finite value.")
        self._mass = value

    @property
    def mmoi(self) -> np.ndarray:
        return self._mmoi

    @mmoi.setter
    def mmoi(self, value: np.ndarray):
        if value.shape != (3, 3):
            raise ValueError("Moment of inertia matrix must be a 3x3 numpy array.")
        if np.isnan(value).any() or (value < 0).any():
            raise ValueError(
                "Moment of inertia matrix must not contain negative or NaN values."
            )
        self._mmoi = value

    @property
    def frame(self) -> Frame:
        """Returns the frame associated with the body."""
        return self._frame

    @property
    def origin(self) -> Point:
        """Returns the origin point of the body."""
        return self._origin

    @property
    def points(self) -> List[Point]:
        """
        Returns a list of points associated with the body, starting with the origin point.
        """
        return [self.origin] + self._points

    @property
    def points_dict(self) -> dict[str, Point]:
        """
        Returns a dictionary of points associated with the body, where the keys are point names, and the values are the points themselves.
        """
        return {point.name: point for point in self.points}

    # ----------
    # Public methods
    # ----------
    def add_points(self, *points: Point):
        """
        Add points to the body.

        Args:
            points
                Points to be added to the body.
        """
        # Validate each point
        for point in points:
            if not isinstance(point, Point):
                raise TypeError("All points must be instances of Point.")
            if point.frame.name != self.name:
                raise ValueError(
                    f"Point '{point.name}' must be in the '{self.name}' frame."
                )

        self._points += points

    def add_points_from_array(self, points: np.ndarray):
        """
        Add points to the body from a numpy array.

        Args:
            points
                A 2D numpy array where each row represents a point in the form [x, y, z].
        """
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Points must be a 2D numpy array with shape (n, 4).")

        new_points = [
            Point(name=point[0], frame=self.origin.frame, position=point[1:])
            for point in points
        ]
        self.add_points(*new_points)

    def add_points_from_dict(self, points: dict):
        """
        Add points to the body from a dictionary, where the keys are the point names.

        Args:
            points
                A dictionary where keys are point names and values are 3D positions.
        """
        if not isinstance(points, dict):
            raise TypeError("Points must be a dictionary.")

        new_points = [
            Point(name=name, frame=self.origin.frame, position=np.array(position))
            for name, position in points.items()
        ]
        self.add_points(*new_points)
