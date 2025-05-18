import math
import pytest

from scalar import Scalar
from nn import NN
from helpers import constants

TOL = constants.get("TOL")


class TestNNForward:
    # Ieșirea are dimensiunea egală cu ultimul layer specificat
    @pytest.mark.parametrize(
        "dims, x",
        [([3, 4, 2], [0.1, -0.2, 0.5]),
         ([2, 2, 1], [1.0, 0.5])],
        ids=["3-4-2", "2-2-1"]
    )
    def test_output_shape(self, dims, x):
        net = NN(dims)
        y = net(x)
        if isinstance(y, Scalar):
            assert dims[-1] == 1
        else:
            assert len(y) == dims[-1]

    # Toate ieșirile trebuie să fie Scalar
    def test_output_type_scalar(self):
        net = NN([2, 3, 2])
        out = net([0.3, -0.7])
        assert all(isinstance(v, Scalar) for v in out)

    # Datorită activării tanh, rezultatele ies în (-1,1)
    def test_tanh_range_final(self):
        net = NN([3, 5, 5, 2])
        out = net([1.0, -2.0, 0.5])
        assert all(-1.0 < v.valoare < 1.0 for v in out)

    # Rețea mică cu ponderi/bias specificate produce valoarea așteptată
    def test_forward_manual_values(self):
        net = NN([2, 1])  # un singur neuron
        # setăm manual ponderi și bias
        neuron = net.layers[0].neuroni[0]
        neuron.ponderi = [Scalar(2.0), Scalar(-1.0)]
        neuron.bias = Scalar(0.5)

        x = [1.5, -2.0]
        y = net(x)
        expected = math.tanh(2.0 * 1.5 + (-1.0) * -2.0 + 0.5)
        assert math.isclose(y.valoare, expected, rel_tol=TOL, abs_tol=TOL)
