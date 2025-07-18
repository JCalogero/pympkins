from dataclasses import dataclass, field


@dataclass
class NameMixin:
    """
    A mixin class to handle the name attribute for objects.
    """
    _name: str = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if self._name is not None:
            raise ValueError("Name cannot be changed after initialization.")
        if not isinstance(value, str):
            raise TypeError("Name must be a string.")
        self._name = value
