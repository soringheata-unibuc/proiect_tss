import pytest

from nn import NN
from scalar import Scalar
from helpers import constants

TOL = constants.get("TOL")


class TestNNCore:
    # parametri() – număr corect și referință directă
    def test_parametri_len_and_identity(self):
        dims = [3, 4, 2]
        net = NN(dims)
        params = net.parametri()
        expected_len = sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1))
        assert len(params) == expected_len
        # modificăm un parametru și verificăm propagarea referinței
        params[0].valoare += 1.0
        first_weight = net.layers[0].neuroni[0].ponderi[0]
        assert first_weight.valoare == params[0].valoare

    # reset_deriv() setează toate gradientele la zero
    def test_reset_deriv_sets_all_zero(self):
        net = NN([2, 3, 1])
        loss = (net([0.4, -0.6]) - 0.2) ** 2
        loss.retroprop()
        assert any(p.derivata != 0.0 for p in net.parametri())
        net.reset_deriv()
        assert all(p.derivata == 0.0 for p in net.parametri())

    # Ultimul strat cu 1 neuron produce Scalar, nu listă
    def test_scalar_output_when_single_neuron(self):
        net = NN([2, 3, 1])
        out = net([0.1, 0.2])
        assert isinstance(out, Scalar)

    # Ultimul strat cu k neuroni produce listă de lungime k
    @pytest.mark.parametrize("k", [2, 5], ids=["k=2", "k=5"])
    def test_forward_shape_multi_output(self, k):
        net = NN([3, 4, k])
        out = net([0.3, -0.1, 0.7])
        assert isinstance(out, list) and len(out) == k

    # repr conține lanțul straturilor
    def test_repr_contains_layers_chain(self):
        net = NN([2, 2, 2])
        rep = repr(net)
        assert "Layer(" in rep and "->" in rep

    # Două apeluri forward produc noduri Scalar diferite (grafuri independente)
    def test_forward_calls_independent(self):
        net = NN([2, 3, 1])
        out1 = net([0.5, -0.5])
        out2 = net([0.5, -0.5])
        assert out1 is not out2 and out1.valoare == out2.valoare
