"""Visualization helpers for synchronization experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def set_plot_style() -> None:
    """Consistent, publication-friendly style for all figures."""

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save figure and close to keep memory usage low in large sweeps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_r_vs_time(times: np.ndarray, r_t: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, r_t, color="#1f77b4", lw=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Order parameter r(t)")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    return fig


def plot_r_smoothing_comparison(
    times: np.ndarray,
    r_low: np.ndarray,
    r_high: np.ndarray,
    smooth_low: np.ndarray,
    smooth_high: np.ndarray,
    window_steps: int,
) -> plt.Figure:
    """Compare raw and smoothed coherence traces for low/high coupling."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    axes[0].plot(times, r_low, color="#9ecae1", lw=1.4, alpha=0.9, label="Raw r(t)")
    axes[0].plot(times, smooth_low, color="#08519c", lw=2.2, label=f"Moving avg ({window_steps} steps)")
    axes[0].set_title("Low K: persistent incoherent fluctuations")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Order parameter r(t)")
    axes[0].legend(fontsize=9)

    axes[1].plot(times, r_high, color="#fdd0a2", lw=1.4, alpha=0.9, label="Raw r(t)")
    axes[1].plot(times, smooth_high, color="#a63603", lw=2.2, label=f"Moving avg ({window_steps} steps)")
    axes[1].set_title("High K: stable synchronized plateau")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.set_ylim(0.0, 1.02)

    return fig


def plot_steady_r_vs_k(k_values: np.ndarray, r_ss: np.ndarray, k_c: float | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(k_values, r_ss, marker="o", lw=2, color="#d62728")
    if k_c is not None and np.isfinite(k_c):
        ax.axvline(k_c, color="black", ls="--", lw=1.5, label=f"Estimated Kc={k_c:.2f}")
        ax.legend()
    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Steady-state coherence <r>")
    ax.set_title("Bifurcation / phase transition: coherence vs coupling")
    ax.set_ylim(0.0, 1.02)
    return fig


def plot_frequency_bifurcation(k_values: np.ndarray, omega_eff_matrix: np.ndarray) -> plt.Figure:
    """Classical bifurcation-style plot: asymptotic frequencies vs coupling K."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, coupling_strength in enumerate(k_values):
        ax.scatter(
            np.full(omega_eff_matrix.shape[1], coupling_strength),
            omega_eff_matrix[idx],
            s=8,
            alpha=0.35,
            color="#1f77b4",
            linewidths=0,
        )
    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Asymptotic effective frequencies Omega_i")
    ax.set_title("Bifurcation diagram: frequency branches vs coupling")
    return fig


def plot_sync_time_vs_k(k_values: np.ndarray, t_sync: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(k_values, t_sync, marker="s", lw=2, color="#2ca02c")
    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel(f"Time to reach r > {threshold:.2f}")
    ax.set_title("Synchronization speed vs coupling strength")
    return fig


def plot_correlation_matrix(corr: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", ax=ax, cbar_kws={"label": "C_ij"})
    ax.set_title("Local phase correlation matrix C_ij = cos(theta_i - theta_j)")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    return fig


def plot_correlation_distance(distances: np.ndarray, corr_mean: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(distances, corr_mean, marker="o", lw=2, color="#9467bd")
    ax.set_xlabel("Shortest-path distance")
    ax.set_ylabel("Mean C_ij")
    ax.set_title("Phase correlation decay with graph distance")
    return fig


def plot_frequency_locking(omega: np.ndarray, omega_eff: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(omega, omega_eff, s=28, alpha=0.8, color="#ff7f0e")
    lims = [min(np.min(omega), np.min(omega_eff)), max(np.max(omega), np.max(omega_eff))]
    ax.plot(lims, lims, "k--", lw=1.3, label="Omega = omega")
    ax.set_xlabel("Natural frequency omega_i")
    ax.set_ylabel("Effective frequency Omega_i")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_psd_comparison(freq_low: np.ndarray, psd_low: np.ndarray, freq_high: np.ndarray, psd_high: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(freq_low, psd_low, lw=2, color="#8c564b", label="Incoherent regime")
    ax.plot(freq_high, psd_high, lw=2, color="#17becf", label="Synchronized regime")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power spectral density")
    ax.set_yscale("log")
    ax.set_title("Power spectrum comparison")
    ax.legend()
    return fig


def plot_phase_heatmap(theta: np.ndarray, times: np.ndarray, title: str) -> plt.Figure:
    wrapped = np.mod(theta, 2 * np.pi)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(
        wrapped.T,
        cmap="twilight",
        ax=ax,
        cbar_kws={"label": "Phase (wrapped to [0, 2pi))"},
    )
    tick_idx = np.linspace(0, len(times) - 1, 6, dtype=int)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels([f"{times[idx]:.1f}" for idx in tick_idx])
    ax.set_xlabel("Time")
    ax.set_ylabel("Oscillator index")
    ax.set_title(title)
    return fig


def plot_phase_heatmap_comparison(theta_low: np.ndarray, theta_high: np.ndarray, times: np.ndarray) -> plt.Figure:
    """Side-by-side heatmaps for incoherent and synchronized regimes."""

    wrapped_low = np.mod(theta_low, 2 * np.pi)
    wrapped_high = np.mod(theta_high, 2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    im0 = axes[0].imshow(wrapped_low.T, origin="lower", aspect="auto", cmap="twilight", vmin=0, vmax=2 * np.pi)
    im1 = axes[1].imshow(wrapped_high.T, origin="lower", aspect="auto", cmap="twilight", vmin=0, vmax=2 * np.pi)

    tick_idx = np.linspace(0, len(times) - 1, 6, dtype=int)
    tick_labels = [f"{times[idx]:.1f}" for idx in tick_idx]
    for ax, title in zip(axes, ["Low K", "High K"]):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Time")
        ax.set_title(f"Phase heatmap ({title})")
    axes[0].set_ylabel("Oscillator index")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Phase (wrapped to [0, 2pi))")
    _ = im0  # keep consistent references and avoid lint warnings in some environments
    return fig


def plot_phase_raster(theta: np.ndarray, times: np.ndarray, title: str) -> plt.Figure:
    wrapped = np.mod(theta, 2 * np.pi)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    n_times, n_nodes = wrapped.shape
    sample_t = np.linspace(0, n_times - 1, min(400, n_times), dtype=int)
    for node in range(n_nodes):
        ax.scatter(times[sample_t], wrapped[sample_t, node], s=2, alpha=0.35, color="#1f77b4")
    ax.set_xlabel("Time")
    ax.set_ylabel("Phase")
    ax.set_title(title)
    ax.set_ylim(0, 2 * np.pi)
    return fig


def plot_network_phase(adjacency: np.ndarray, phases: np.ndarray, title: str) -> plt.Figure:
    graph = nx.from_numpy_array(adjacency)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    pos = nx.spring_layout(graph, seed=7)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=np.mod(phases, 2 * np.pi),
        cmap="twilight",
        node_size=90,
        ax=ax,
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=0.8, ax=ax)
    plt.colorbar(nodes, ax=ax, label="Phase")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_phase_histograms(hist_data: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 2 * np.pi, 30)
    for label, phase in hist_data.items():
        ax.hist(np.mod(phase, 2 * np.pi), bins=bins, alpha=0.45, density=True, label=label)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Density")
    ax.set_title("Phase distribution evolution")
    ax.legend()
    return fig


def plot_topology_comparison(k_values: np.ndarray, topology_results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for label, metrics in topology_results.items():
        axes[0].plot(k_values, metrics["r_ss"], marker="o", lw=2, label=label)
        axes[1].plot(k_values, metrics["t_sync"], marker="s", lw=2, label=label)

    axes[0].set_title("Final coherence vs K")
    axes[0].set_ylabel("<r>")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_title("Synchronization time vs K")
    axes[1].set_ylabel("t_sync")
    for ax in axes:
        ax.set_xlabel("K")
        ax.legend(fontsize=9)

    return fig


def plot_rewiring_effect(p_values: np.ndarray, sync_time: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(p_values, sync_time, marker="o", lw=2, color="#bcbd22")
    ax.set_xscale("log")
    ax.set_xlabel("Rewiring probability p (log scale)")
    ax.set_ylabel("Synchronization time")
    ax.set_title("A few long-range links can sharply accelerate synchronization")
    return fig


def plot_coherence_fluctuations(
    times: np.ndarray,
    std_low: np.ndarray,
    std_high: np.ndarray,
    tail_std_low: float,
    tail_std_high: float,
) -> plt.Figure:
    """Plot short-time fluctuation amplitude of r(t) for low/high coupling."""

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(times, std_low, color="#1f77b4", lw=2, label=f"Low K (tail std={tail_std_low:.3f})")
    ax.plot(times, std_high, color="#d62728", lw=2, label=f"High K (tail std={tail_std_high:.3f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Local std of r(t)")
    ax.set_title("Fluctuation-based stability of synchronization")
    ax.legend()
    return fig


def plot_plv_matrix(plv: np.ndarray, threshold: float) -> plt.Figure:
    """Heatmap of phase-locking value matrix PLV_ij."""

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    sns.heatmap(plv, vmin=0.0, vmax=1.0, cmap="viridis", ax=ax, cbar_kws={"label": "PLV_ij"})
    ax.set_title(f"Late-time phase-locking matrix (threshold={threshold:.2f})")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    return fig


def plot_cluster_sizes(cluster_sizes: np.ndarray, threshold: float) -> plt.Figure:
    """Bar plot of connected synchronized-cluster sizes."""

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    indices = np.arange(1, cluster_sizes.size + 1)
    ax.bar(indices, cluster_sizes, color="#6a3d9a", alpha=0.85)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size")
    ax.set_title(f"Partial synchronization clusters from PLV >= {threshold:.2f}")
    ax.set_xticks(indices)
    return fig


def plot_perturbation_recovery(
    times: np.ndarray,
    r_reference: np.ndarray,
    r_perturbed: np.ndarray,
    threshold: float,
    recovery_time: float,
) -> plt.Figure:
    """Compare synchronized continuation against perturbed restart."""

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    ax.plot(times, r_reference, lw=2.2, color="#2ca02c", label="Continuation from synchronized state")
    ax.plot(times, r_perturbed, lw=2.2, color="#ff7f0e", label="After phase perturbation")
    ax.axhline(threshold, color="black", ls="--", lw=1.2, label=f"Threshold r={threshold:.2f}")
    if np.isfinite(recovery_time):
        ax.axvline(recovery_time, color="#9467bd", ls=":", lw=1.4, label=f"Recovery time={recovery_time:.2f}")
    ax.set_xlabel("Time after perturbation")
    ax.set_ylabel("Order parameter r(t)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Robustness: resynchronization after finite perturbation")
    ax.legend(fontsize=9)
    return fig


def plot_finite_size(n_values: np.ndarray, r_values: np.ndarray, t_values: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(n_values, r_values, marker="o", lw=2, color="#1f77b4")
    axes[0].set_xlabel("System size N")
    axes[0].set_ylabel("Steady-state <r>")
    axes[0].set_title("Finite-size effect on coherence")
    axes[0].set_ylim(0.0, 1.02)

    axes[1].plot(n_values, t_values, marker="s", lw=2, color="#d62728")
    axes[1].set_xlabel("System size N")
    axes[1].set_ylabel("Synchronization time")
    axes[1].set_title("Finite-size effect on speed")
    return fig
