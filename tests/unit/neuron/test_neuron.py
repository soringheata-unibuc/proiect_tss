import math
import statistics
import pytest

from helpers import numeric_grad, constants
from neuron import Neuron
from scalar import Scalar

TOL = constants.get('TOL')


class TestNeuron:
    # Forward determinist cu ponderi și bias setați manual
    def test_forward_determinist(self):
        n = Neuron(intrari=3)
        n.ponderi = [Scalar(1.0), Scalar(-2.0), Scalar(0.5)]
        n.bias = Scalar(0.25)
        x = [Scalar(2.0), Scalar(3.0), Scalar(-1.0)]
        y = n(x)
        expected = math.tanh(1.0 * 2.0 + (-2.0) * 3.0 + 0.5 * (-1.0) + 0.25)
        assert pytest.approx(y.valoare, rel=TOL) == expected

    # Ieșirea tanh trebuie să rămână în intervalul (-1, 1)
    @pytest.mark.parametrize(
        "vec",
        [[-1.0, -1.0, -1.0], [0.0, 0.2, -0.3], [5.0, -7.0, 1.0]],
        ids=["neg", "mix", "large"],
    )
    def test_tanh_range(self, vec):
        n = Neuron(intrari=3)
        y = n([Scalar(v) for v in vec])
        assert -1.0 < y.valoare < 1.0

    # Gradientele ponderilor: autograd ≈ finite-difference
    def test_gradient_ponderi(self):
        n = Neuron(intrari=2)
        x = [Scalar(0.6), Scalar(-1.2)]

        def expr():
            return n(x)

        out = expr()
        for p in n.parametri():
            p.derivata = 0.0
        out.retroprop()

        for i, w in enumerate(n.ponderi):
            num = numeric_grad(expr, w)
            assert math.isclose(w.derivata, num, rel_tol=1e-3, abs_tol=1e-3), f"w{i}"

    # Gradientul bias-ului: autograd ≈ finite-difference
    def test_gradient_bias(self):
        n = Neuron(intrari=1)
        x = [Scalar(0.9)]

        def expr():
            return n(x)

        out = expr()
        n.bias.derivata = 0.0
        out.retroprop()
        num = numeric_grad(expr, n.bias)
        assert math.isclose(n.bias.derivata, num, rel_tol=1e-3, abs_tol=1e-3)

    # Lista de parametri conține exact n_inputs + 1 elemente
    def test_parametri_len_and_identity(self):
        n = Neuron(intrari=4)
        params = n.parametri()
        assert len(params) == 5
        params[0].valoare += 1.0
        assert n.ponderi[0].valoare == params[0].valoare

    # Lungime input diferită de numărul de ponderi -> ValueError
    def test_input_length_mismatch(self):
        n = Neuron(intrari=3)
        with pytest.raises(ValueError):
            _ = n([Scalar(1.0), Scalar(2.0)])

    # Inițializarea He: deviația standard a ponderilor ≈ sqrt(2 / fan_in)
    def test_initializare_he_dispersion(self):
        m = 1000
        intrari = 3
        valori = []
        for _ in range(m):
            n = Neuron(intrari)
            valori.extend([w.valoare for w in n.ponderi])
        sigma_emp = statistics.pstdev(valori)
        sigma_teo = math.sqrt(2.0 / intrari)
        assert abs(sigma_emp - sigma_teo) < 0.2 * sigma_teo
