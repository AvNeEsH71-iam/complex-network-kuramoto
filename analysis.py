"""Analysis functions for synchronization, locking, and spectral diagnostics."""

from __future__ import annotations

import networkx as nx
import numpy as np

from dynamics import effective_frequencies, simulate_kuramoto


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """Compute Kuramoto order parameter r(t) from phase trajectory theta[t, i]."""

    return np.abs(np.mean(np.exp(1j * theta), axis=1))


def moving_average(signal: np.ndarray, window: int = 21) -> np.ndarray:
    """Centered moving average used to visualize coherence trends."""

    if window < 1:
        raise ValueError("window must be >= 1")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def local_fluctuation_std(signal: np.ndarray, window: int = 21) -> np.ndarray:
    """Sliding standard deviation to quantify short-time synchronization stability."""

    mean = moving_average(signal, window=window)
    mean_sq = moving_average(signal**2, window=window)
    variance = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(variance)


def coherence_stability(
    r_t: np.ndarray,
    window: int = 21,
    tail_fraction: float = 0.3,
) -> dict[str, np.ndarray | float]:
    """Return smoothed coherence and fluctuation diagnostics."""

    smooth = moving_average(r_t, window=window)
    local_std = local_fluctuation_std(r_t, window=window)
    n_tail = max(1, int(np.ceil(tail_fraction * r_t.size)))

    return {
        "smooth": smooth,
        "local_std": local_std,
        "tail_mean": float(np.mean(r_t[-n_tail:])),
        "tail_std": float(np.std(r_t[-n_tail:])),
    }


def steady_state_value(signal: np.ndarray, tail_fraction: float = 0.2) -> float:
    """Return average over the final tail_fraction of a time series."""

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    n_tail = max(1, int(np.ceil(tail_fraction * signal.size)))
    return float(np.mean(signal[-n_tail:]))


def time_to_threshold(times: np.ndarray, signal: np.ndarray, threshold: float = 0.9) -> float:
    """First time at which signal crosses threshold; NaN if never reached."""

    idx = np.where(signal >= threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float(times[idx[0]])


def phase_correlation_matrix(theta_final: np.ndarray) -> np.ndarray:
    """Pairwise local phase correlation C_ij = cos(theta_i - theta_j)."""

    phase_diff = theta_final[:, None] - theta_final[None, :]
    return np.cos(phase_diff)


def correlation_vs_distance(corr: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average C_ij over all pairs with the same graph distance."""

    if corr.shape != dist.shape:
        raise ValueError("corr and dist must have the same shape")

    valid = np.isfinite(dist) & (dist > 0)
    dist_vals = np.unique(dist[valid]).astype(int)

    means = []
    for d in dist_vals:
        mask = valid & np.isclose(dist, d)
        means.append(float(np.mean(corr[mask])))
    return dist_vals, np.array(means, dtype=float)


def power_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided power spectral density via FFT."""

    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(x)
    psd = (np.abs(spectrum) ** 2) / n
    return freqs, psd


def sweep_coupling(
    adjacency: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    k_values: np.ndarray,
    dt: float,
    t_max: float,
    threshold: float = 0.9,
) -> dict[str, np.ndarray]:
    """Sweep coupling K and compute synchronization diagnostics per value."""

    r_ss = np.zeros_like(k_values, dtype=float)
    t_sync = np.full_like(k_values, np.nan, dtype=float)

    for idx, coupling_strength in enumerate(k_values):
        run = simulate_kuramoto(
            adjacency=adjacency,
            omega=omega,
            theta0=theta0,
            coupling_strength=float(coupling_strength),
            dt=dt,
            t_max=t_max,
        )
        r_t = order_parameter(run["theta"])
        r_ss[idx] = steady_state_value(r_t)
        t_sync[idx] = time_to_threshold(run["t"], r_t, threshold=threshold)

    return {"K": k_values, "r_ss": r_ss, "t_sync": t_sync}


def critical_coupling(k_values: np.ndarray, r_ss: np.ndarray, rise: float = 0.2) -> float:
    """Heuristic critical coupling where steady coherence starts to rise."""

    baseline_count = max(2, int(np.ceil(0.15 * k_values.size)))
    baseline = float(np.mean(r_ss[:baseline_count]))
    trigger = baseline + rise
    idx = np.where(r_ss >= trigger)[0]
    if idx.size == 0:
        return float("nan")
    return float(k_values[idx[0]])


def locking_data(
    adjacency: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    coupling_strength: float,
    dt: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return natural and effective frequencies for a fixed K."""

    run = simulate_kuramoto(adjacency, omega, theta0, coupling_strength, dt, t_max)
    omega_eff = effective_frequencies(run["theta"], run["t"])
    return omega, omega_eff


def frequency_bifurcation_data(
    adjacency: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    k_values: np.ndarray,
    dt: float,
    t_max: float,
    start_fraction: float = 0.5,
) -> np.ndarray:
    """Build a bifurcation-like matrix of effective frequencies vs coupling K.

    Returns an array of shape (len(k_values), n_nodes) where each row contains
    the asymptotic effective frequencies Omega_i for one coupling value.
    """

    omega_eff_matrix = np.zeros((k_values.size, omega.size), dtype=float)
    for idx, coupling_strength in enumerate(k_values):
        run = simulate_kuramoto(
            adjacency=adjacency,
            omega=omega,
            theta0=theta0,
            coupling_strength=float(coupling_strength),
            dt=dt,
            t_max=t_max,
        )
        omega_eff_matrix[idx] = effective_frequencies(
            run["theta"],
            run["t"],
            start_fraction=start_fraction,
        )

    return omega_eff_matrix


def phase_locking_matrix(theta: np.ndarray, start_fraction: float = 0.6) -> np.ndarray:
    """Compute PLV_ij = |<exp(i(theta_i-theta_j))>| over the late-time window."""

    if theta.ndim != 2:
        raise ValueError("theta must have shape (time, nodes)")

    start_idx = int(np.floor(start_fraction * (theta.shape[0] - 1)))
    tail = theta[start_idx:]
    phase_diff = tail[:, :, None] - tail[:, None, :]
    return np.abs(np.mean(np.exp(1j * phase_diff), axis=0))


def synchronization_clusters(
    theta: np.ndarray,
    threshold: float = 0.97,
    start_fraction: float = 0.6,
) -> dict[str, np.ndarray | float]:
    """Detect partially synchronized groups from late-time phase-locking values."""

    plv = phase_locking_matrix(theta, start_fraction=start_fraction)
    lock_adj = (plv >= threshold).astype(int)
    np.fill_diagonal(lock_adj, 0)

    graph = nx.from_numpy_array(lock_adj)
    cluster_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    if not cluster_sizes:
        cluster_sizes = [0]

    n_nodes = theta.shape[1]
    return {
        "plv": plv,
        "cluster_sizes": np.array(cluster_sizes, dtype=int),
        "largest_cluster_fraction": float(cluster_sizes[0]) / float(n_nodes),
        "threshold": float(threshold),
    }
