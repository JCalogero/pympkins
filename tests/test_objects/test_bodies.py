import pytest
import numpy as np
from hypothesis import given, strategies as st

from pympkins.objects.bodies import Frame, Point, Body

# ----------
# Frame tests
# ----------


def test_frame_creation():
    f = Frame(name="test_frame", fixed=True)
    assert f.name == "test_frame"
    assert f.fixed is True


# ----------
# Point tests
# ----------
def test_point_creation_defaults():
    p = Point(name="P")
    assert p.name == "P"
    assert isinstance(p.position, np.ndarray)
    assert p.position.shape == (3,)


def test_point_creation_with_custom_frame_and_position():
    frame = Frame(name="custom_frame", fixed=False)
    pos = np.array([1.0, 2.0, 3.0])
    p = Point(name="Pcustom", frame=frame, position=pos)
    assert p.name == "Pcustom"
    assert p.frame.name == "custom_frame"
    assert np.array_equal(p.position, pos)


# ----------
# Body tests
# ----------
def test_body_creation_and_origin():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    assert b.name == "B"
    assert b.mass == 1.0
    assert np.array_equal(b.mmoi, np.eye(3))
    assert isinstance(b.origin, Point)
    assert b.origin.frame.name == "B"


def test_body_mass_validation():
    with pytest.raises(ValueError):
        Body(name="B", mass=0, mmoi=np.eye(3))


def test_body_mmoi_validation():
    with pytest.raises(ValueError):
        Body(name="B", mass=1.0, mmoi=np.ones((2, 2)))


def test_add_points_type_and_frame_validation():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    p_good = Point(name="P1", frame=Frame(name="B"))
    p_bad = Point(name="P2", frame=Frame(name="other"))
    with pytest.raises(ValueError):
        b.add_points(p_bad)
    with pytest.raises(TypeError):
        b.add_points("not_a_point")
    b.add_points(p_good)
    assert p_good in b.points


def test_add_points_from_array_shape():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    arr = np.array([["P1", 1, 2, 3], ["P2", 4, 5, 6]], dtype=object)
    b.add_points_from_array(arr)
    assert any(p.name == "P1" for p in b.points)
    with pytest.raises(ValueError):
        b.add_points_from_array(np.array([[1, 2, 3]]))


def test_add_points_from_dict_type():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    d = {"P1": [1, 2, 3], "P2": [4, 5, 6]}
    b.add_points_from_dict(d)
    assert any(p.name == "P1" for p in b.points)
    with pytest.raises(TypeError):
        b.add_points_from_dict([("P1", [1, 2, 3])])


def test_body_points_origin_is_first():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    p1 = Point(name="P1", frame=b.frame)
    p2 = Point(name="P2", frame=b.frame)
    b.add_points(p1, p2)
    points = b.points
    assert points[0] is b.origin
    assert [p.name for p in points] == [b.origin.name, "P1", "P2"]


def test_body_points_dict_property():
    b = Body(name="B", mass=1.0, mmoi=np.eye(3))
    p1 = Point(name="P1", frame=b.frame)
    p2 = Point(name="P2", frame=b.frame)
    b.add_points(p1, p2)
    points_dict = b.points_dict
    assert isinstance(points_dict, dict)
    assert set(points_dict.keys()) == {b.origin.name, "P1", "P2"}
    assert points_dict[b.origin.name] is b.origin
    assert points_dict["P1"] is p1
    assert points_dict["P2"] is p2


@given(
    st.text(min_size=1),
    st.floats(allow_nan=True, allow_infinity=False),
    st.lists(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=False), min_size=3, max_size=3
        ),
        min_size=3,
        max_size=3,
    ),
)
def test_body_hypothesis(name, mass, mmoi_list):
    mmoi = np.array(mmoi_list)
    # Check for invalid mass
    if not np.isfinite(mass) or mass <= 0:
        with pytest.raises(ValueError):
            Body(name=name, mass=mass, mmoi=mmoi)
        return
    # Check for invalid mmoi (shape or NaN)
    if mmoi.shape != (3, 3) or np.isnan(mmoi).any() or np.less(mmoi, 0).any():
        with pytest.raises(ValueError):
            Body(name=name, mass=mass, mmoi=mmoi)
        return
    # Valid case
    b = Body(name=name, mass=mass, mmoi=mmoi)
    assert b.mass == mass
    assert np.array_equal(b.mmoi, mmoi)
