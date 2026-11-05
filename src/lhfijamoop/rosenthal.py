"""Rosenthal moving point source, an analytic stand-in for a thermal solver.

It is used to verify the chain and to prototype inverse design without a cluster run. It
is not a model of a real melt pool. Everything is written so that complex arguments work,
which lets a complex step supply parameter and time derivatives to machine precision.
"""

from dataclasses import dataclass

import numpy as np

from .solidification import ThermalJet

_H = 1e-30


@dataclass(frozen=True)
class Material:
    """Thermal constants in SI units."""

    k: float = 25.0
    alpha: float = 6.0e-6
    T0: float = 300.0
    absorptivity: float = 0.75


def temperature(point, t, power, speed, mat: Material):
    """Temperature at point (x, y, z) and time t. The source starts at x = 0 at t = 0."""
    x, y, z = point
    xi = x - speed * t
    r = (xi * xi + y * y + z * z) ** 0.5
    return mat.T0 + mat.absorptivity * power / (2 * np.pi * mat.k * r) * np.exp(
        -speed * (xi + r) / (2 * mat.alpha)
    )


def gradient(point, t, power, speed, mat: Material):
    """Analytic spatial gradient of the temperature."""
    x, y, z = point
    xi = x - speed * t
    r = (xi * xi + y * y + z * z) ** 0.5
    rise = temperature(point, t, power, speed, mat) - mat.T0
    a = speed / (2 * mat.alpha)
    return np.array(
        [
            rise * (-a * (1 + xi / r) - xi / (r * r)),
            rise * (-a * y / r - y / (r * r)),
            rise * (-a * z / r - z / (r * r)),
        ]
    )


def dT_dt(point, t, power, speed, mat: Material):
    """Time derivative of the temperature at a fixed point."""
    return -speed * gradient(point, t, power, speed, mat)[0]


def liquidus_crossing(point, power, speed, mat: Material, T_liquidus: float) -> float:
    """Return the cooling crossing time of T_liquidus by bisection.

    The peak at a fixed point lies behind the source. The search brackets the crossing
    between that peak and a time far behind it.
    """
    x = point[0]
    distance = (point[1] ** 2 + point[2] ** 2) ** 0.5
    times = x / speed + np.linspace(-2.0, 200.0, 4000) * distance / speed
    values = np.array([temperature(point, tq, power, speed, mat) for tq in times])
    peak = int(np.argmax(values))
    if values[peak] <= T_liquidus:
        raise ValueError("this point never reaches the liquidus")
    after = np.nonzero(values[peak:] < T_liquidus)[0]
    if after.size == 0:
        raise ValueError("this point does not cool through the liquidus in the time window")
    low, high = times[peak + after[0] - 1], times[peak + after[0]]
    for _ in range(200):
        mid = 0.5 * (low + high)
        if temperature(point, mid, power, speed, mat) > T_liquidus:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def jet(point, t, power, speed, mat: Material) -> ThermalJet:
    """Build the thermal data for s = (log power, log speed) at time t."""

    def cplx(fn, ds_power=0.0, ds_speed=0.0, dt=0.0):
        return fn(
            point,
            t + 1j * dt,
            power * np.exp(1j * ds_power),
            speed * np.exp(1j * ds_speed),
            mat,
        )

    grad = np.real(gradient(point, t, power, speed, mat))
    T_t = float(np.real(dT_dt(point, t, power, speed, mat)))
    T_tt = float(np.imag(cplx(dT_dt, dt=_H)) / _H)
    grad_t = np.imag(cplx(gradient, dt=_H)) / _H
    T_s = np.array(
        [
            np.imag(cplx(temperature, ds_power=_H)) / _H,
            np.imag(cplx(temperature, ds_speed=_H)) / _H,
        ]
    )
    T_ts = np.array(
        [
            np.imag(cplx(dT_dt, ds_power=_H)) / _H,
            np.imag(cplx(dT_dt, ds_speed=_H)) / _H,
        ]
    )
    grad_s = np.column_stack(
        [
            np.imag(cplx(gradient, ds_power=_H)) / _H,
            np.imag(cplx(gradient, ds_speed=_H)) / _H,
        ]
    )
    return ThermalJet(T_t, T_tt, grad, grad_t, T_s, T_ts, grad_s)
