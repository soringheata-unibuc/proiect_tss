import pytest

from scalar import Scalar


@pytest.fixture(
    params=[(-3.0, 5.0), (0.0, 0.0), (1.5, -2.5), (1e3, -1e3)],
    ids=lambda p: f"a={p[0]}_b={p[1]}"
)
def scalar_pair(request):
    a, b = request.param
    return Scalar(a), Scalar(b)
