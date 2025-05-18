from scalar import Scalar
from neuron import Neuron
from nn import NN


class TestSmoke:
    def test_smoke_constructors(self):
        Scalar(0.1)
        n = Neuron(3)
        assert len(n.ponderi) == 3
        NN([2, 3, 1])

    def test_smoke_forward(self):
        net = NN([2, 3, 1])
        out = net([0.0, 0.0])  # forward-pass simplu
        # doar verificăm tipul și că se poate accesa valoarea scalară
        assert isinstance(out, Scalar)
        float(out.valoare)  # conversie
