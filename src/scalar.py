import math
from typing import Callable, Iterable, Self


class Scalar:
    def __init__(
        self,
        valoare: float,
        parinti: Iterable[Self] = (),
        operatie: str = '',
    ) -> None:
        if math.isnan(valoare) or math.isinf(valoare):
            raise ValueError("Valoarea nu poate fi NaN sau inf")

        self.valoare: float = float(valoare)
        self.derivata: float = 0.0
        self._parinti: set[Self] = set(parinti)
        self._operatie: str = operatie
        self._retro: Callable[[float], None] = lambda g: None

    # Adunare
    def __add__(self, alt: Self | float) -> Self:
        alt = alt if isinstance(alt, Scalar) else Scalar(alt)
        suma = Scalar(self.valoare + alt.valoare, (self, alt), '+')

        def calc_grad(g: float) -> None:
            self.derivata += g
            alt.derivata += g

        suma._retro = calc_grad
        return suma

    # Adunare inversă
    def __radd__(self, alt: float) -> Self:
        return self + alt

    # Negativ
    def __neg__(self) -> Self:
        return self * -1.0

    # Scădere
    def __sub__(self, alt: Self | float) -> Self:
        return self + (-alt)

    # Multiplicare
    def __mul__(self, alt: Self | float) -> Self:
        alt = alt if isinstance(alt, Scalar) else Scalar(alt)
        produs = Scalar(self.valoare * alt.valoare, (self, alt), '*')

        def calc_grad(g: float) -> None:
            self.derivata += alt.valoare * g
            alt.derivata += self.valoare * g

        produs._retro = calc_grad
        return produs

    # Multiplicare inversă
    def __rmul__(self, alt: float) -> Self:
        return self * alt

    # Împărțire
    def __truediv__(self, alt: Self | float) -> Self:
        alt = alt if isinstance(alt, Scalar) else Scalar(alt)
        return self * alt ** -1  # delegăm la pow

    # Împărțire inversă
    def __rtruediv__(self, alt: float) -> Self:
        return Scalar(alt) / self

    # Exponențiere
    def __pow__(self, exp: float) -> Self:
        # domeniu valid: bază ≥0 sau exponent întreg
        if self.valoare < 0 and not float(exp).is_integer():
            raise ValueError("Negative base with non-integer exponent not supported")

        if self.valoare == 0.0 and exp < 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")

        rez = Scalar(self.valoare ** exp, (self,), f'**{exp}')

        def calc_grad(g: float) -> None:
            self.derivata += exp * (self.valoare ** (exp - 1)) * g

        rez._retro = calc_grad
        return rez

    # Activări element-wise
    def relu(self) -> Self:
        rez = Scalar(self.valoare if self.valoare > 0 else 0.0, (self,), 'ReLU')

        def calc_grad(g: float) -> None:
            self.derivata += (1.0 if self.valoare > 0 else 0.0) * g

        rez._retro = calc_grad
        return rez

    def tanh(self) -> Self:
        t = math.tanh(self.valoare)
        rez = Scalar(t, (self,), 'tanh')

        def calc_grad(g: float) -> None:
            self.derivata += (1.0 - t * t) * g

        rez._retro = calc_grad
        return rez

    # Propagare înapoi
    def retroprop(self) -> None:
        """Pornește retropropagarea (setează dL/dself = 1)."""
        ordine: list[Scalar] = []
        vizitat: set[Scalar] = set()

        def construieste_noduri(nod: Scalar) -> None:
            if nod not in vizitat:
                vizitat.add(nod)
                for p in nod._parinti:
                    construieste_noduri(p)
                ordine.append(nod)

        construieste_noduri(self)
        self.derivata = 1.0

        for nod in reversed(ordine):
            nod._retro(nod.derivata)

    def __repr__(self) -> str:
        return f'Scalar(valoare={self.valoare:.4f}, deriv={self.derivata:.4f})'
