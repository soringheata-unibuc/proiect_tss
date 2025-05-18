import math
import operator
import pytest

from scalar import Scalar
from helpers import constants

TOL = constants.get("TOL")


class TestScalarRobustness:
    # Constructorul trebuie să respingă NaN şi ±inf
    @pytest.mark.parametrize(
        "val",
        [float("nan"), float("inf"), float("-inf")],
        ids=["NaN", "inf", "-inf"],
    )
    def test_init_nonfinite_raises(self, val):
        with pytest.raises(ValueError):
            _ = Scalar(val)

    # Operaţii care duc la rezultate non-finite → ValueError
    @pytest.mark.parametrize(
        "lhs_val, rhs_val, op",
        [
            (1e308, 2.0, operator.mul),  # overflow: inf
            (1.0, float("nan"), operator.add),  # NaN propagare
        ],
        ids=["overflow_mul", "nan_add"],
    )
    def test_nonfinite_operation_raises(self, lhs_val, rhs_val, op):
        a = Scalar(lhs_val)
        with pytest.raises(ValueError):
            if op is operator.add:
                _ = op(a, rhs_val)  # rhs e float, devine Scalar(float('nan')) în interior
            else:
                _ = op(a, Scalar(rhs_val))

    # Underflow controlat: rezultatul este 0.0, nu NaN
    @pytest.mark.parametrize("small", [1e-200, 1e-250], ids=["1e-200", "1e-250"])
    def test_mul_underflow_small_values(self, small):
        c = Scalar(small) * Scalar(small)
        assert c.valoare == 0.0 and not math.isnan(c.valoare)

    # reset_deriv curăţă gradientele între două retropropagări
    def test_reset_deriv_clears_gradients(self):
        x = Scalar(2.0)
        ((x * 3) + 4).retroprop()
        assert x.derivata != 0.0
        x.derivata = 0.0
        assert x.derivata == 0.0
        ((x * 5) + 1).retroprop()
        assert x.derivata != 0.0

    # repr(Scalar) trebuie să conţină câmpurile valoare şi deriv
    @pytest.mark.parametrize("val", [3.14, -7.0], ids=["pozitiv", "negativ"])
    def test_repr_contains_keywords(self, val):
        rep = repr(Scalar(val))
        assert "valoare=" in rep and "deriv=" in rep
