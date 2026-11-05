import unittest

from lhfijamoop.recipe import Recipe, predict, source_perturbation


class SourcePerturbation(unittest.TestCase):
    def test_nominal_recipe_has_no_change(self):
        nominal = Recipe(285, 960, 67)
        change = source_perturbation(nominal, nominal, 0.5)
        self.assertTrue(all(value == 0 for value in change.values()))

    def test_power_moves_depth_radius_through_the_exponent(self):
        change = source_perturbation(Recipe(285 * 1.1, 960, 67), Recipe(285, 960, 67), 0.5)
        self.assertAlmostEqual(change["Rz"] / change["P"], 0.5)
        self.assertEqual(change["Rx"], 0.0)

    def test_spot_moves_both_surface_radii_and_depth(self):
        change = source_perturbation(Recipe(285, 960, 67 * 1.2), Recipe(285, 960, 67), 1.0)
        self.assertAlmostEqual(change["Rx"], change["Ry"])
        self.assertAlmostEqual(change["Rz"], -1.5 * change["Rx"])

    def test_prediction_is_linear(self):
        value = predict(10.0, {"P": 2.0, "Vx": -1.0}, {"P": 0.5, "Vx": 0.25})
        self.assertAlmostEqual(value, 10.75)


if __name__ == "__main__":
    unittest.main()
