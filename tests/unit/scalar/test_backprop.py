import math

from helpers import numeric_grad, constants
from scalar import Scalar

TOL = constants.get("TOL")


class TestScalarBackprop:
    # f(x) = ((3x)+2)^2  →  df/dx = 2*(3x+2)*3
    def test_chain_rule_single_path(self):
        x = Scalar(1.0)
        f = ((x * 3) + 2) ** 2
        f.retroprop()
        expected = 2 * (3 * 1.0 + 2) * 3  # 30
        assert math.isclose(x.derivata, expected, rel_tol=TOL, abs_tol=TOL)

    # Diamond graph: f = a*b + a*b -> df/da = 2*b, df/db = 2*a
    def test_diamond_graph_accumulation(self):
        a, b = Scalar(2.0), Scalar(-3.0)
        f = a * b + a * b
        f.retroprop()
        assert math.isclose(a.derivata, 2 * b.valoare, rel_tol=TOL, abs_tol=TOL)
        assert math.isclose(b.derivata, 2 * a.valoare, rel_tol=TOL, abs_tol=TOL)

    # Ordonarea topologică rămâne stabilă: două rulări dau aceleaşi gradiente
    def test_topological_order_stable(self):
        x = Scalar(1.0)

        def build_graph():
            a = x * 2
            b = a + 3
            c = b * 4
            return c ** 2

        g1 = build_graph()
        g1.retroprop()
        grad1 = x.derivata

        x.derivata = 0.0
        g2 = build_graph()
        g2.retroprop()
        grad2 = x.derivata

        assert math.isclose(grad1, grad2, rel_tol=TOL, abs_tol=TOL)

    # Frunzele nefolosite rămân cu gradient zero după retro-propagare
    def test_multiple_leaves_gradient_zero(self):
        a, b = Scalar(2.0), Scalar(3.0)
        d = Scalar(5.0)  # frunză neconectată
        c = a * b
        f = c ** 2
        f.retroprop()
        assert a.derivata != 0.0 and b.derivata != 0.0
        assert d.derivata == 0.0

    # Gradient numeric vs autograd pe g(x,y)=tanh(x*y)
    def test_grad_numeric_close(self):
        x, y = Scalar(0.5), Scalar(-1.2)

        def expr():
            return (x * y).tanh()

        out = expr()
        for p in (x, y):
            p.derivata = 0.0
        out.retroprop()

        for param in (x, y):
            num = numeric_grad(expr, param)
            assert math.isclose(param.derivata, num, rel_tol=1e-3, abs_tol=1e-3)
