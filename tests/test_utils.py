import pytest
from hypothesis import given, strategies as st
from pympkins.pympkins.utils import NameMixin


def test_name_set_once():
    obj = NameMixin("Alice")
    assert obj.name == "Alice"
    with pytest.raises(ValueError):
        obj.name = "Bob"


def test_name_type_error():
    with pytest.raises(TypeError):
        NameMixin(123)


@given(st.text())
def test_name_accepts_string(s):
    obj = NameMixin(s)
    assert obj.name == s


@given(st.one_of(st.integers(), st.floats(), st.none(), st.lists(st.integers())))
def test_name_rejects_non_string(val):
    with pytest.raises(TypeError):
        NameMixin(val)
