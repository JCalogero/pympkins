
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class Joint:
    """
    Represents a joint in the pympkins framework.

    Args:
        name
            The name of the joint.
        
    """
    name: str
