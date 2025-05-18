import math
import pytest

from helpers import numeric_grad, constants
from scalar import Scalar

TOL = constants.get('TOL')


class TestScalarOps:
    # Adunare / Scădere / Negativ
    def test_add_sub_neg(self, scalar_pair):
        a, b = scalar_pair

        # add
        c = a + b
        assert pytest.approx(c.valoare, abs=TOL) == a.valoare + b.valoare

        # radd (float + Scalar)
        d = 2.0 + a
        assert pytest.approx(d.valoare, abs=TOL) == 2.0 + a.valoare

        # sub
        e = a - b
        assert pytest.approx(e.valoare, abs=TOL) == a.valoare - b.valoare

        # neg
        f = -a
        assert pytest.approx(f.valoare, abs=TOL) == -a.valoare

    # Multiplicare
    def test_mul_div(self, scalar_pair):
        a, b = scalar_pair

        # mul
        c = a * b
        assert pytest.approx(c.valoare, abs=TOL) == a.valoare * b.valoare

        # rmul   (float * Scalar)
        d = 3.0 * a
        assert pytest.approx(d.valoare, abs=TOL) == 3.0 * a.valoare

    # Împărțire – valori valide (fără zero)
    @pytest.mark.parametrize(
        "a_val,b_val",
        [(-3.0, 5.0), (1.5, -2.5), (1e3, -1e3)],
        ids=["-3/5", "1.5/-2.5", "1000/-1000"],
    )
    def test_div_valide(self, a_val, b_val):
        a, b = Scalar(a_val), Scalar(b_val)
        assert (a / b).valoare == pytest.approx(a_val / b_val, rel=TOL)
        # rtruediv  (float / Scalar)
        assert (10.0 / a).valoare == pytest.approx(10.0 / a_val, rel=TOL)

    # Împărţire a / 0 –> excepţie
    @pytest.mark.parametrize(
        "a_val",
        [-3.0, 0.0, 1.5],
        ids=["-3_over_0", "0_over_0", "1.5_over_0"],
    )
    def test_div_zero_denominator(self, a_val):
        a, b = Scalar(a_val), Scalar(0.0)
        with pytest.raises(ZeroDivisionError):
            _ = a / b

    # împărţire 10 / a  cu a = 0 –> excepţie
    @pytest.mark.parametrize(
        "a_val",
        [0.0],
        ids=["10_over_0"],
    )
    def test_rtruediv_zero(self, a_val):
        a = Scalar(a_val)
        with pytest.raises(ZeroDivisionError):
            _ = 10.0 / a

    # Exponențiere - Cazuri VALIDE
    @pytest.mark.parametrize(
        "a_val, exp",
        [
            (3.0, 2),  # pozitiv, întreg
            (-4.0, 2),  # negativ, exponent întreg
            (1.5, 0.5),  # pozitiv, fracţie
        ],
        ids=["3^2", "(-4)^2", "1.5^0.5"],
    )
    def test_pow_valori_corecte(self, a_val, exp):
        a = Scalar(a_val)
        rez = a ** exp
        assert pytest.approx(rez.valoare, rel=TOL) == a_val ** exp

    # Exponențiere - Cazuri INVALIDE (excepții)
    @pytest.mark.parametrize(
        "a_val, exp, exc",
        [
            (0.0, -1, ZeroDivisionError),  # 0 la putere negativă
            (-3.0, 0.5, ValueError),  # bază negativă + exponent fracție
        ],
        ids=["0^-1", "(-3)^0.5"],
    )
    def test_pow_exceptii(self, a_val, exp, exc):
        a = Scalar(a_val)
        with pytest.raises(exc):
            _ = a ** exp

    # Gradient-check global (add + mul)
    def test_grad_matches_numeric(self):
        """
        f(w) = ((w * 3) + 4)**2
        Analitic: df/dw = 2*((w*3)+4) * 3
        """
        w = Scalar(1.2)

        def expr():
            return ((w * 3) + 4) ** 2

        # autograd
        out = expr()
        w.derivata = 0.0
        out.retroprop()
        auto_g = w.derivata

        # numeric
        num_g = numeric_grad(expr, w)

        assert math.isclose(auto_g, num_g, rel_tol=TOL, abs_tol=TOL)
