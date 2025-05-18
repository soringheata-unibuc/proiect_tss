from scalar import Scalar

constants = dict(
    TOL=1e-6,  # toleranța în comparații
)


def numeric_grad(expr_fn, param: Scalar, eps: float = 1e-4) -> float:
    """
    Aproximarea ∂expr/∂param folosind diferențe finite centralizate.
    `expr_fn` primește *obiectul* `param` modificat, întoarce Scalar-ul expresiei.

    Args:
        expr_fn:
        param:
        eps:

    """
    original = param.valoare

    # f(w+eps)
    param.valoare = original + eps
    plus = expr_fn().valoare

    # f(w-eps)
    param.valoare = original - eps
    minus = expr_fn().valoare

    # restore
    param.valoare = original

    return (plus - minus) / (2 * eps)
