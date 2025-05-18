import math
import random
from typing import Sequence

from scalar import Scalar


class Neuron:
    """Perceptron cu activare ReLU/tanh și inițializare He."""

    def __init__(self, intrari: int) -> None:
        sigma = math.sqrt(2.0 / intrari)
        self.ponderi: list[Scalar] = [
            Scalar(random.randint(-1, 1)) for _ in range(intrari)
            # Scalar(random.gauss(0.0, sigma)) for _ in range(intrari)
        ]
        self.bias: Scalar = Scalar(0.0)

    def __call__(self, x: Sequence[Scalar]) -> Scalar:
        if len(x) != len(self.ponderi):
            raise ValueError(
                f"Lungime input {len(x)} diferită de numărul de ponderi {len(self.ponderi)}"
            )

        s: Scalar = self.bias
        for w, xi in zip(self.ponderi, x):
            s += w * xi

        # return s.relu()
        return s.tanh()

    def parametri(self) -> list[Scalar]:
        return [*self.ponderi, self.bias]

    def __repr__(self) -> str:
        return f'Neuron({len(self.ponderi)} ponderi)'
