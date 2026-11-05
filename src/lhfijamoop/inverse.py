"""Find the recipe that reaches a target microstructure descriptor set."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Result:
    """Result of an inverse solve.

    Attributes:
        s: final design variables.
        residual: final residual in log descriptors.
        iterations: Gauss-Newton iterations used.
        converged: True when the residual norm reached the tolerance.
    """

    s: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool


def gauss_newton(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    s0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_step: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> Result:
    """Solve log q(s) = log target by bounded, damped Gauss-Newton.

    Args:
        evaluate: returns descriptors q, shape (m,), and dq/ds, shape (m, n).
        target: positive target descriptors, shape (m,).
        s0: starting design, shape (n,).
        lower: lower bounds on s.
        upper: upper bounds on s.
        max_step: largest step in the infinity norm.
        tolerance: residual norm for convergence.
        max_iterations: iteration limit.

    Returns:
        The final design and its residual. A target outside the reachable set returns
        with converged set to False, with the closest design the iteration found.
    """
    s = np.clip(np.asarray(s0, dtype=float), lower, upper)
    target = np.asarray(target, dtype=float)
    q, jac = evaluate(s)
    residual = np.log(q / target)
    for iteration in range(max_iterations):
        norm = float(np.linalg.norm(residual))
        if norm < tolerance:
            return Result(s, residual, iteration, True)
        step = np.linalg.lstsq(jac / q[:, None], -residual, rcond=None)[0]
        largest = float(np.max(np.abs(step)))
        if largest > max_step:
            step *= max_step / largest
        scale = 1.0
        while True:
            trial = np.clip(s + scale * step, lower, upper)
            try:
                q_trial, jac_trial = evaluate(trial)
                residual_trial = np.log(q_trial / target)
                accepted = np.linalg.norm(residual_trial) < norm
            except ValueError:
                accepted = False
            if accepted:
                s, q, jac, residual = trial, q_trial, jac_trial, residual_trial
                break
            scale *= 0.5
            if scale < 1e-6:
                return Result(s, residual, iteration, False)
    return Result(s, residual, max_iterations, bool(np.linalg.norm(residual) < tolerance))
