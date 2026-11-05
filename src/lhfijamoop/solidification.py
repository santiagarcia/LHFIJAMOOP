"""Solidification descriptors and their parameter derivatives.

A fixed material point crosses the liquidus at time t_c(s), defined implicitly by
T(x, t_c; s) = T_L. The implicit function theorem gives

    dt_c/ds = -T_s / T_t

Descriptors are evaluated on that moving crossing, so their total derivatives carry a
t_c term as well as the direct parameter term:

    G      = |grad T|                 thermal gradient
    Tdot   = -T_t                     cooling rate
    R      = Tdot / G                 solidification front speed

Inputs are plain arrays, so any thermal solver that can export T, its space and time
derivatives, and their parameter derivatives can drive this module.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalJet:
    """Thermal field data at one point and time.

    Attributes:
        T_t: time derivative of T, a scalar.
        T_tt: second time derivative of T, a scalar.
        grad: spatial gradient of T, shape (3,).
        grad_t: time derivative of the gradient, shape (3,).
        T_s: parameter derivative of T, shape (n,).
        T_ts: parameter derivative of T_t, shape (n,).
        grad_s: parameter derivative of the gradient, shape (3, n).
    """

    T_t: float
    T_tt: float
    grad: np.ndarray
    grad_t: np.ndarray
    T_s: np.ndarray
    T_ts: np.ndarray
    grad_s: np.ndarray


@dataclass(frozen=True)
class Descriptors:
    """Descriptors at the crossing, with derivatives of shape (n,)."""

    G: float
    Tdot: float
    R: float
    dt_ds: np.ndarray
    dG_ds: np.ndarray
    dTdot_ds: np.ndarray
    dR_ds: np.ndarray


def descriptors(jet: ThermalJet) -> Descriptors:
    """Evaluate G, cooling rate and R, and their total parameter derivatives."""
    if jet.T_t == 0.0:
        raise ValueError("T_t is zero at the crossing, so the crossing time is not defined")
    dt_ds = -np.asarray(jet.T_s, dtype=float) / jet.T_t
    grad = np.asarray(jet.grad, dtype=float)
    G = float(np.linalg.norm(grad))
    if G == 0.0:
        raise ValueError("the thermal gradient is zero, so R is not defined")
    unit = grad / G
    # Total derivative of the gradient along the moving crossing.
    dgrad = np.asarray(jet.grad_s, dtype=float) + np.outer(jet.grad_t, dt_ds)
    dG_ds = unit @ dgrad
    Tdot = -float(jet.T_t)
    dTdot_ds = -(np.asarray(jet.T_ts, dtype=float) + jet.T_tt * dt_ds)
    R = Tdot / G
    dR_ds = dTdot_ds / G - Tdot * dG_ds / G**2
    return Descriptors(G, Tdot, R, dt_ds, dG_ds, dTdot_ds, dR_ds)
