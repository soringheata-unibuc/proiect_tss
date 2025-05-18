import math
import pytest

from nn import NN
from helpers import constants, numeric_grad

TOL = constants.get("TOL")


class TestNNBackward:
    # Gradienții devin nenuli după retropropagare pe o pierdere MSE
    def test_backward_nonzero_grads(self):
        net = NN([3, 4, 1])
        x = [0.2, -0.4, 1.0]
        target = 0.8

        y_pred = net(x)  # Scalar
        loss = (y_pred - target) ** 2
        loss.retroprop()

        grads = [p.derivata for p in net.parametri()]
        assert any(g != 0.0 for g in grads)

    # Grad autograd ≈ grad numeric pentru prima pondere şi bias
    @pytest.mark.parametrize("param_index", [0, 1], ids=["w0", "b0"])
    def test_numeric_gradient_close(self, param_index):
        net = NN([2, 3, 1])
        x = [0.5, -1.2]
        target = -0.3

        params = net.parametri()
        param_to_check = params[param_index]  # indice 0 = prima pondere, 1 = bias primul neuron

        def expr():
            pred = net(x)
            return (pred - target) ** 2  # Scalar loss

        loss = expr()
        for p in params:
            p.derivata = 0.0
        loss.retroprop()

        num_grad = numeric_grad(expr, param_to_check)
        assert math.isclose(param_to_check.derivata, num_grad, rel_tol=1e-3, abs_tol=1e-3)

    # După reset, gradientul se recalculează identic (fără acumulare eronată)
    def test_gradients_reset_correct(self):
        net = NN([2, 2, 1])
        x = [1.0, -0.5]
        target = 0.1

        def expr():
            return (net(x) - target) ** 2

        # prima rulare
        expr().retroprop()
        grad_first = net.parametri()[0].derivata

        # reset
        for p in net.parametri():
            p.derivata = 0.0

        # a doua rulare
        expr().retroprop()
        grad_second = net.parametri()[0].derivata

        assert math.isclose(grad_first, grad_second, rel_tol=TOL, abs_tol=TOL)
