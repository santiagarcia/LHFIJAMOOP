"""First-order prediction of melt quantities across recipes from one nominal run.

The solver exports derivatives with respect to multiplicative perturbations s_i of each
input, evaluated at s_i = 1, which equal d q / d ln p_i.

The heat source geometry may depend on the recipe. Its surface radii follow the spot
size, and its depth radius is taken as a power n of the normalised enthalpy P / sqrt(v D^3),
so a recipe change moves the depth radius as well, and the total derivative needs that
term. The exponent is an input. Prediction is linear in the log of each recipe variable.
"""

from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class Recipe:
    """Printing recipe. Units only need to be consistent across recipes."""

    power: float
    speed: float
    spot: float


def source_perturbation(recipe: Recipe, nominal: Recipe, depth_exponent: float) -> dict:
    """Return the log change of each exported input for a change of recipe.

    Args:
        recipe: the new recipe.
        nominal: the recipe of the nominal run.
        depth_exponent: exponent n in the depth scaling of the heat source.

    Returns:
        Log changes keyed by the solver parameter names P, Vx, Rx, Ry and Rz.
    """
    d_power = log(recipe.power / nominal.power)
    d_speed = log(recipe.speed / nominal.speed)
    d_spot = log(recipe.spot / nominal.spot)
    d_depth = depth_exponent * (d_power - 0.5 * d_speed - 1.5 * d_spot)
    return {"P": d_power, "Vx": d_speed, "Rx": d_spot, "Ry": d_spot, "Rz": d_depth}


def predict(nominal_value: float, sensitivity: dict, perturbation: dict) -> float:
    """Return the first-order prediction of a quantity at a perturbed recipe."""
    return nominal_value + sum(sensitivity[name] * d for name, d in perturbation.items())
