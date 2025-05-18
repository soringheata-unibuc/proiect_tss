import statistics

import pytest

from helpers import constants
from nn import NN

TOL = constants.get("TOL")


class TestTrainingStep:
    # Un singur pas: pierderea MSE trebuie să scadă
    def test_loss_decreases_one_step(self):
        net = NN([3, 4, 1])
        x = [0.3, -0.8, 0.5]
        target = 0.7

        pred_before = net(x)
        loss_before = (pred_before - target) ** 2

        net.reset_deriv()
        loss_before.retroprop()
        lr = 0.05
        for p in net.parametri():
            p.valoare -= lr * p.derivata

        pred_after = net(x)
        loss_after = (pred_after - target) ** 2

        assert loss_after.valoare < loss_before.valoare

    # Cel puțin un parametru se modifică după actualizare
    def test_parameters_updated(self):
        net = NN([2, 2, 1])
        x = [1.0, -0.5]
        target = -0.2

        loss = (net(x) - target) ** 2
        net.reset_deriv()
        loss.retroprop()

        params = net.parametri()
        old_values = [p.valoare for p in params]

        for p in params:
            p.valoare -= 0.1 * p.derivata

        assert any(p.valoare != old for p, old in zip(params, old_values))

    # După reset, gradientele se golesc toate
    def test_gradients_zeroed_after_step(self):
        net = NN([2, 3, 1])
        x = [0.4, -0.3]
        target = 0.05
        loss = (net(x) - target) ** 2
        loss.retroprop()
        for p in net.parametri():
            p.derivata = 0.0
        assert all(p.derivata == 0.0 for p in net.parametri())

    # Pierderea are trend descrescător pe 10 pași
    def test_multiple_steps_loss_trend(self):
        net = NN([3, 5, 1])
        x = [0.1, -0.7, 0.9]
        target = -0.4
        lr = 0.05
        losses = []

        for _ in range(10):
            pred = net(x)
            loss = (pred - target) ** 2
            losses.append(loss.valoare)
            net.reset_deriv()
            loss.retroprop()
            for p in net.parametri():
                p.valoare -= lr * p.derivata

        assert statistics.mean(losses[-3:]) < statistics.mean(losses[:3])

    # Cu lr=0.0 parametrii nu se schimbă și pierderea rămâne identică
    def test_learning_rate_effect(self):
        net = NN([2, 2, 1])
        x = [0.8, -0.1]
        target = 0.3

        loss_before = (net(x) - target) ** 2
        net.reset_deriv()
        loss_before.retroprop()

        old_params = [p.valoare for p in net.parametri()]
        for p in net.parametri():
            p.valoare -= 0.0 * p.derivata

        loss_after = (net(x) - target) ** 2
        assert loss_after.valoare == pytest.approx(loss_before.valoare, rel=TOL)
        assert all(p.valoare == old for p, old in zip(net.parametri(), old_params))

    # Pe un singur exemplu, după 300 de pași, MSE devine foarte mic (<1e-4)
    def test_overfit_single_example(self):
        net = NN([2, 4, 1])
        x = [0.6, -0.9]
        target = -0.5
        lr = 0.05

        for _ in range(300):
            loss = (net(x) - target) ** 2
            net.reset_deriv()
            loss.retroprop()
            for p in net.parametri():
                p.valoare -= lr * p.derivata

        final_loss = (net(x) - target) ** 2
        assert final_loss.valoare < 1e-4
