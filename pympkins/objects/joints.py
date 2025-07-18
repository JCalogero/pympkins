"""
This file contains the basic body classes.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .bodies import Body


@dataclass
class Joint:
    """
    Represents a joint in the pympkins framework.

    Args:
        name
            The name of the joint.
        
    """
    name: str
    body_i: Body
    point_i: str
    body_j: Body
    point_j: str
