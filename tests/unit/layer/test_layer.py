import math
import pytest

from scalar import Scalar
from layer import Layer
from helpers import numeric_grad, constants

TOL = constants.get('TOL')


class TestLayer:
    # Forward: ieșirea are exact n_neuroni elemente
    @pytest.mark.parametrize("n_in,n_out", [(4, 6), (3, 1)], ids=["4x6", "3x1"])
    def test_forward_shape(self, n_in, n_out):
        layer = Layer(n_in, n_out)
        x = [Scalar(0.1 * i) for i in range(n_in)]
        y = layer(x)
        assert len(y) == n_out

    # Forward: fiecare ieșire este Scalar
    def test_forward_scalar_types(self):
        layer = Layer(3, 5)
        x = [Scalar(0.2), Scalar(-0.4), Scalar(1.0)]
        y = layer(x)
        assert all(isinstance(v, Scalar) for v in y)

    # Forward determinist: valori așteptate cu ponderi & bias controlate
    def test_forward_values_determinist(self):
        layer = Layer(2, 2)
        # setăm manual ponderi & bias pentru ambii neuroni
        w_sets = [[1.0, -1.0], [0.5, 0.5]]
        b_sets = [0.0, 0.25]
        for n, (w_vec, b_val) in zip(layer.neuroni, zip(w_sets, b_sets)):
            n.ponderi = [Scalar(w) for w in w_vec]
            n.bias = Scalar(b_val)
        x = [Scalar(2.0), Scalar(-3.0)]
        y = layer(x)
        expected = [
            math.tanh(1.0 * 2.0 + (-1.0) * -3.0 + 0.0),
            math.tanh(0.5 * 2.0 + 0.5 * -3.0 + 0.25),
        ]
        assert pytest.approx(y[0].valoare, rel=TOL) == expected[0]
        assert pytest.approx(y[1].valoare, rel=TOL) == expected[1]

    # Lista parametri: trebuie să conțină (n_in + 1) · n_out elemente
    def test_parametri_len(self):
        n_in, n_out = 3, 4
        layer = Layer(n_in, n_out)
        params = layer.parametri()
        assert len(params) == (n_in + 1) * n_out
        params[0].valoare += 1.0
        assert layer.neuroni[0].ponderi[0].valoare == params[0].valoare

    # Gradientele propagate corespund gradientelor numerice (un neuron ales)
    def test_gradients_propagate(self):
        n_in = 3
        layer = Layer(n_in, 2)
        x = [Scalar(0.3), Scalar(-0.7), Scalar(1.1)]

        def expr():
            # folosim doar ieșirea primului neuron
            return layer(x)[0]

        out = expr()
        for p in layer.parametri():
            p.derivata = 0.0
        out.retroprop()

        # verificăm gradient numeric pentru ponderile neuronului 0
        for w in layer.neuroni[0].ponderi + [layer.neuroni[0].bias]:
            num = numeric_grad(expr, w)
            assert math.isclose(w.derivata, num, rel_tol=1e-3, abs_tol=1e-3)

    # Input cu lungime greșită → ValueError
    def test_input_length_mismatch(self):
        layer = Layer(3, 2)
        x = [Scalar(1.0), Scalar(2.0)]  # doar 2 intrări
        with pytest.raises(ValueError):
            _ = layer(x)

    # Repr trebuie să includă "Layer(<n>×Neuron)"
    def test_repr_contine_dimensiuni(self):
        layer = Layer(2, 5)
        assert "Layer(5×Neuron)" in repr(layer)
