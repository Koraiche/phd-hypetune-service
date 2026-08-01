import numpy as np

from phase2_search.parameter_space import CategoricalParam, FloatParam, IntParam, ParameterSpace


def test_int_param_rounds_and_clips():
    space = ParameterSpace([IntParam("threads", 1, 4)])
    assert space.decode(np.array([2.6]))["threads"] == 3
    assert space.decode(np.array([-5.0]))["threads"] == 1
    assert space.decode(np.array([100.0]))["threads"] == 4


def test_float_param_clips_without_rounding():
    space = ParameterSpace([FloatParam("temp", 0.0, 1.0)])
    assert space.decode(np.array([0.37]))["temp"] == 0.37
    assert space.decode(np.array([5.0]))["temp"] == 1.0


def test_categorical_param_boundaries():
    space = ParameterSpace([CategoricalParam("kv", ["f16", "q8_0", "q4_0"])])
    assert space.decode(np.array([0.0]))["kv"] == "f16"
    assert space.decode(np.array([0.9999]))["kv"] == "f16"
    assert space.decode(np.array([1.0]))["kv"] == "q8_0"
    assert space.decode(np.array([2.0]))["kv"] == "q4_0"
    # ub is nudged just below len(choices); decoding exactly at ub must not
    # raise an index error.
    assert space.decode(np.array([space.ub[0]]))["kv"] == "q4_0"


def test_mixed_space_dim_and_bounds():
    space = ParameterSpace([
        IntParam("threads", 1, 4),
        CategoricalParam("kv", ["f16", "q8_0", "q4_0"]),
    ])
    assert space.dim == 2
    np.testing.assert_allclose(space.lb, [1.0, 0.0])
    assert space.ub[0] == 4.0
    assert 2.0 < space.ub[1] < 3.0
