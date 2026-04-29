"""Network utilities for Kuramoto synchronization experiments.

This module provides a clean implementation of the Watts-Strogatz network model
and helper functions used by the analysis and visualization layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class NetworkSpec:
	"""Container for network construction parameters."""

	n_nodes: int
	mean_degree: int = 6
	rewiring_prob: float = 0.1
	seed: int | None = None


def generate_watts_strogatz(spec: NetworkSpec) -> np.ndarray:
	"""Generate an undirected Watts-Strogatz adjacency matrix.

	The construction follows the original idea: start from a ring lattice and
	rewire each clockwise edge with probability p, avoiding self-loops and
	duplicate edges.
	"""

	n = spec.n_nodes
	k = spec.mean_degree
	p = spec.rewiring_prob

	if n < 4:
		raise ValueError("n_nodes must be >= 4")
	if k <= 0 or k >= n:
		raise ValueError("mean_degree must satisfy 0 < k < n")
	if k % 2 != 0:
		raise ValueError("mean_degree must be even for a ring-lattice WS graph")
	if not (0.0 <= p <= 1.0):
		raise ValueError("rewiring_prob must be in [0, 1]")

	rng = np.random.default_rng(spec.seed)

	neighbors = [set() for _ in range(n)]
	half_k = k // 2

	# Ring lattice: each node connects to k/2 neighbors on each side.
	for i in range(n):
		for d in range(1, half_k + 1):
			j = (i + d) % n
			neighbors[i].add(j)
			neighbors[j].add(i)

	# Rewire each forward edge (i, i+d) with probability p.
	for i in range(n):
		for d in range(1, half_k + 1):
			j = (i + d) % n
			if j not in neighbors[i]:
				continue
			if rng.random() >= p:
				continue

			neighbors[i].remove(j)
			neighbors[j].remove(i)

			forbidden = neighbors[i].copy()
			forbidden.add(i)
			candidates = np.array([node for node in range(n) if node not in forbidden])
			if candidates.size == 0:
				neighbors[i].add(j)
				neighbors[j].add(i)
				continue

			new_j = int(rng.choice(candidates))
			neighbors[i].add(new_j)
			neighbors[new_j].add(i)

	adjacency = np.zeros((n, n), dtype=float)
	for i in range(n):
		for j in neighbors[i]:
			adjacency[i, j] = 1.0

	adjacency = np.maximum(adjacency, adjacency.T)
	np.fill_diagonal(adjacency, 0.0)
	return adjacency


def adjacency_to_graph(adjacency: np.ndarray) -> nx.Graph:
	"""Convert a NumPy adjacency matrix to a NetworkX graph."""

	return nx.from_numpy_array(adjacency)


def shortest_path_matrix(adjacency: np.ndarray) -> np.ndarray:
	"""Compute pairwise shortest-path distances for an undirected graph."""

	graph = adjacency_to_graph(adjacency)
	lengths = dict(nx.all_pairs_shortest_path_length(graph))
	n = adjacency.shape[0]
	dist = np.full((n, n), np.inf, dtype=float)
	for i in range(n):
		for j, d_ij in lengths[i].items():
			dist[i, j] = float(d_ij)
	return dist


def degree_sequence(adjacency: np.ndarray) -> np.ndarray:
	"""Return node degrees from a binary adjacency matrix."""

	return np.sum(adjacency > 0, axis=1).astype(int)


def ring_lattice_matches_neighbors(adjacency: np.ndarray, neighbors_per_side: int = 3) -> bool:
	"""Check if each node is connected exactly to +/-1..+/-neighbors_per_side on a ring."""

	n_nodes = adjacency.shape[0]
	for i in range(n_nodes):
		expected = {
			(i + d) % n_nodes for d in range(1, neighbors_per_side + 1)
		} | {
			(i - d) % n_nodes for d in range(1, neighbors_per_side + 1)
		}
		actual = set(np.where(adjacency[i] > 0)[0].tolist())
		if actual != expected:
			return False
	return True


def adjacency_checks(adjacency: np.ndarray, expected_mean_degree: int | None = None) -> dict[str, float | bool]:
	"""Return basic structural checks used in report validation tables."""

	degrees = degree_sequence(adjacency)
	symmetric = bool(np.allclose(adjacency, adjacency.T))
	self_loops = bool(np.any(np.diag(adjacency) != 0))

	checks: dict[str, float | bool] = {
		"symmetric": symmetric,
		"has_self_loops": self_loops,
		"degree_min": float(np.min(degrees)),
		"degree_mean": float(np.mean(degrees)),
		"degree_max": float(np.max(degrees)),
	}
	if expected_mean_degree is not None:
		checks["mean_degree_matches_target"] = bool(np.isclose(np.mean(degrees), expected_mean_degree))
	return checks


def topology_label(rewiring_prob: float) -> str:
	"""Human-readable label used in figure legends and reports."""

	if np.isclose(rewiring_prob, 0.0):
		return "Regular lattice (p=0)"
	if np.isclose(rewiring_prob, 1.0):
		return "Random network (p=1)"
	return f"Small-world (p={rewiring_prob:g})"
