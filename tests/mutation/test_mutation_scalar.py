import math

from helpers import constants
from scalar_mutant import Scalar

TOL = constants.get("TOL")


class TestMutationScalar:

    def test_mutant_detect_no_init_derivata(self):
        x = Scalar(1.0)
        f = ((x * 3) + 2) ** 2
        f.retroprop()
        expected = 2 * (3 * 1.0 + 2) * 3  # 30
        assert not math.isclose(x.derivata, expected, rel_tol=TOL, abs_tol=TOL)  # mutant detectat, trece pentru ca asertia este inversata

    def test_mutant_detect_wrong_tanh_grad(self):
        x = Scalar(0.5)

        def expr():
            return x.tanh()

        out = expr()
        x.derivata = 0.0
        out.retroprop()
        expected = 1.0 - math.tanh(0.5) ** 2
        assert not math.isclose(x.derivata, expected, rel_tol=1e-3)  # mutant detectat, trece pentru ca asertia este inversata
