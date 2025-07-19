"""
This file contains the basic body classes.
"""

from dataclasses import dataclass


from .bodies import Body


@dataclass
class Joint:
    """
    Represents a joint in the pympkins framework.

    Args:
        name
            The name of the joint.
        body_a
            The first body connected by the joint.
        point_a
            The point on the first body where the joint is connected.
        body_b
            The second body connected by the joint.
        point_b
            The point on the second body where the joint is connected.
    """

    name: str
    body_a: Body
    point_a: str
    body_b: Body
    point_b: str
