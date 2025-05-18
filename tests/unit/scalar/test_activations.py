import math
import pytest

from scalar import Scalar
from helpers import constants, numeric_grad

TOL = constants.get("TOL")


class TestScalarActivations:
    # ReLU: valoare corectă în domeniile (-), 0, (+)
    @pytest.mark.parametrize("val, expected", [(-2.0, 0.0), (0.0, 0.0), (3.5, 3.5)], ids=["neg", "zero", "pos"])
    def test_relu_values(self, val, expected):
        out = Scalar(val).relu()
        assert math.isclose(out.valoare, expected, abs_tol=TOL)

    # ReLU: gradient 0 pentru x<=0, 1 pentru x>0
    @pytest.mark.parametrize("val, grad_exp", [(-1.0, 0.0), (0.0, 0.0), (2.0, 1.0)], ids=["neg", "zero", "pos"])
    def test_relu_gradients(self, val, grad_exp):
        x = Scalar(val)
        y = x.relu()
        y.retroprop()
        assert math.isclose(x.derivata, grad_exp, abs_tol=TOL)

    # tanh(x) trebuie să rămână în (-1,1) pentru valori diverse
    @pytest.mark.parametrize("val", [-5.0, -1.0, 0.0, 1.0, 5.0], ids=["-5", "-1", "0", "1", "5"])
    def test_tanh_range(self, val):
        out = Scalar(val).tanh().valoare
        assert -1.0 < out < 1.0

    # tanh(-x) == -tanh(x) (simetrie impară)
    def test_tanh_odd_symmetry(self):
        x = 1.23
        pos = Scalar(x).tanh().valoare
        neg = Scalar(-x).tanh().valoare
        assert math.isclose(neg, -pos, abs_tol=TOL)

    # Gradient tanh: autograd ≈ numeric în trei puncte
    @pytest.mark.parametrize("val", [-2.0, 0.3, 4.0], ids=["-2", "0.3", "4"])
    def test_tanh_grad_numeric_close(self, val):
        x = Scalar(val)

        def expr():
            return x.tanh()

        out = expr()
        x.derivata = 0.0
        out.retroprop()
        num = numeric_grad(expr, x)
        assert math.isclose(x.derivata, num, rel_tol=1e-3, abs_tol=1e-3)

    # f(x)=tanh(ReLU(x)): la x<=0 grad 0, la x>0 grad (1-t^2)
    @pytest.mark.parametrize("val", [-1.0, 0.0, 1.5], ids=["neg", "zero", "pos"])
    def test_activation_chain_rule(self, val):
        x = Scalar(val)
        f = x.relu().tanh()
        f.retroprop()

        expected_grad = 0.0 if val <= 0 else (1 - math.tanh(val) ** 2)
        assert math.isclose(x.derivata, expected_grad, rel_tol=TOL, abs_tol=TOL)
