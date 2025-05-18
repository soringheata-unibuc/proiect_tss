from typing import Sequence

from neuron import Neuron
from scalar import Scalar


class Layer:
    """Colecție de neuroni care împart aceeași intrare."""

    def __init__(self, intrari: int, neuroni: int) -> None:
        self.neuroni: list[Neuron] = [
            Neuron(intrari) for _ in range(neuroni)
        ]

    def __call__(self, x: Sequence[Scalar]) -> list[Scalar]:
        return [n(x) for n in self.neuroni]

    def parametri(self) -> list[Scalar]:
        res: list[Scalar] = []
        for n in self.neuroni:
            res.extend(n.parametri())
        return res

    def __repr__(self) -> str:
        return f'Layer({len(self.neuroni)}×Neuron)'
