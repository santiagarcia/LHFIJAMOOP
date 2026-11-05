import unittest

import numpy as np

from lhfijamoop import rosenthal
from lhfijamoop.inverse import gauss_newton
from lhfijamoop.solidification import ThermalJet, descriptors

MAT = rosenthal.Material()
POINT = (1.0e-3, 20e-6, -30e-6)
T_LIQ = 1609.0
NOMINAL = np.log([285.0, 0.96])


def evaluate_values(s):
    power, speed = np.exp(s)
    t = rosenthal.liquidus_crossing(POINT, power, speed, MAT, T_LIQ)
    d = descriptors(rosenthal.jet(POINT, t, power, speed, MAT))
    return t, d


class ChainRuleAgainstFiniteDifferences(unittest.TestCase):
    def test_all_descriptors_match_central_differences(self):
        _, d = evaluate_values(NOMINAL)
        analytic = {"t": d.dt_ds, "G": d.dG_ds, "Tdot": d.dTdot_ds, "R": d.dR_ds}
        step = 1e-5
        for i in range(2):
            plus = NOMINAL.copy()
            minus = NOMINAL.copy()
            plus[i] += step
            minus[i] -= step
            t_plus, d_plus = evaluate_values(plus)
            t_minus, d_minus = evaluate_values(minus)
            numeric = {
                "t": (t_plus - t_minus) / (2 * step),
                "G": (d_plus.G - d_minus.G) / (2 * step),
                "Tdot": (d_plus.Tdot - d_minus.Tdot) / (2 * step),
                "R": (d_plus.R - d_minus.R) / (2 * step),
            }
            for name, value in numeric.items():
                with self.subTest(parameter=i, quantity=name):
                    self.assertAlmostEqual(analytic[name][i] / value, 1.0, places=6)


class RejectsUndefinedCrossings(unittest.TestCase):
    def jet(self, **changes):
        base = dict(
            T_t=-1.0,
            T_tt=0.0,
            grad=np.array([1.0, 0.0, 0.0]),
            grad_t=np.zeros(3),
            T_s=np.zeros(1),
            T_ts=np.zeros(1),
            grad_s=np.zeros((3, 1)),
        )
        base.update(changes)
        return ThermalJet(**base)

    def test_stationary_temperature_is_refused(self):
        with self.assertRaises(ValueError):
            descriptors(self.jet(T_t=0.0))

    def test_zero_gradient_is_refused(self):
        with self.assertRaises(ValueError):
            descriptors(self.jet(grad=np.zeros(3)))


class InverseRecipe(unittest.TestCase):
    def evaluate(self, s):
        _, d = evaluate_values(s)
        return np.array([d.G, d.R]), np.vstack([d.dG_ds, d.dR_ds])

    def test_recovers_the_recipe_that_made_the_target(self):
        truth = np.log([250.0, 1.2])
        target, _ = self.evaluate(truth)
        result = gauss_newton(
            self.evaluate,
            target,
            NOMINAL,
            np.log([100.0, 0.3]),
            np.log([500.0, 3.0]),
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.s, truth, rtol=1e-6)

    def test_unreachable_target_reports_failure(self):
        q, _ = self.evaluate(NOMINAL)
        result = gauss_newton(
            self.evaluate,
            q * np.array([50.0, 50.0]),
            NOMINAL,
            np.log([250.0, 0.8]),
            np.log([320.0, 1.2]),
        )
        self.assertFalse(result.converged)


if __name__ == "__main__":
    unittest.main()
