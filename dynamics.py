"""Dynamical core for coupled phase oscillators (Kuramoto model)."""

from __future__ import annotations

import numpy as np


def edge_index_from_adjacency(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected edge index arrays (src, dst) with src < dst."""

    src, dst = np.where(np.triu(adjacency, k=1) > 0)
    return src.astype(int), dst.astype(int)


def kuramoto_rhs(
    theta: np.ndarray,
    omega: np.ndarray,
    coupling_strength: float,
    edge_index: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute dtheta/dt using edge-based accumulation for speed."""

    src, dst = edge_index
    phase_diff = theta[dst] - theta[src]
    edge_flow = np.sin(phase_diff)

    interaction = np.zeros_like(theta)
    np.add.at(interaction, src, edge_flow)
    np.add.at(interaction, dst, -edge_flow)

    return omega + coupling_strength * interaction


def rk4_step(
    theta: np.ndarray,
    omega: np.ndarray,
    coupling_strength: float,
    edge_index: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> np.ndarray:
    """One Runge-Kutta 4 integration step."""

    k1 = kuramoto_rhs(theta, omega, coupling_strength, edge_index)
    k2 = kuramoto_rhs(theta + 0.5 * dt * k1, omega, coupling_strength, edge_index)
    k3 = kuramoto_rhs(theta + 0.5 * dt * k2, omega, coupling_strength, edge_index)
    k4 = kuramoto_rhs(theta + dt * k3, omega, coupling_strength, edge_index)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_kuramoto(
    adjacency: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    coupling_strength: float,
    dt: float,
    t_max: float,
) -> dict[str, np.ndarray]:
    """Simulate Kuramoto oscillators and return time and phase trajectories."""

    n_steps = int(np.ceil(t_max / dt))
    n_nodes = theta0.size

    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    theta = np.zeros((n_steps + 1, n_nodes), dtype=float)
    theta[0] = theta0

    edge_index = edge_index_from_adjacency(adjacency)
    for step in range(n_steps):
        theta[step + 1] = rk4_step(theta[step], omega, coupling_strength, edge_index, dt)

    return {"t": times, "theta": theta}


def effective_frequencies(theta: np.ndarray, times: np.ndarray, start_fraction: float = 0.5) -> np.ndarray:
    """Estimate mean angular frequency over the second half of the simulation."""

    unwrapped = np.unwrap(theta, axis=0)
    start_idx = int(np.floor(start_fraction * (len(times) - 1)))
    duration = times[-1] - times[start_idx]
    if duration <= 0:
        raise ValueError("Simulation duration must be positive.")
    return (unwrapped[-1] - unwrapped[start_idx]) / duration
