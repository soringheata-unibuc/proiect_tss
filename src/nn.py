from typing import List

from layer import Layer
from scalar import Scalar


class NN:
    """
    Rețea perceptron cu mai multe straturi dense și activare ReLU/tanh.
    Ultimul strat nu are softmax; se poate aplica extern.
    """

    def __init__(self, dimensiuni: List[int]) -> None:
        if len(dimensiuni) < 2:
            raise ValueError('Vectorul dimensiuni trebuie să conțină cel puțin două valori.')
        self.layers: list[Layer] = [
            Layer(dimensiuni[i], dimensiuni[i + 1])
            for i in range(len(dimensiuni) - 1)
        ]

    def __call__(self, valori: List[float]) -> Scalar | list[Scalar]:
        activari: list[Scalar] = [Scalar(v) for v in valori]
        for layer in self.layers:
            activari = layer(activari)

        if isinstance(activari, list) and len(activari) == 1:
            return activari[0]

        return activari

    def parametri(self) -> list[Scalar]:
        p: list[Scalar] = []
        for strat in self.layers:
            p.extend(strat.parametri())
        return p

    def reset_deriv(self) -> None:
        for p in self.parametri():
            p.derivata = 0.0

    def __repr__(self) -> str:
        info = ' -> '.join(str(s) for s in self.layers)
        return f'NN({info})'
