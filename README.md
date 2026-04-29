# Kuramoto Synchronization on Watts-Strogatz Networks

This project simulates and analyzes synchronization in a network of coupled phase oscillators:

dtheta_i/dt = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)

## Features

- Watts-Strogatz network generation with configurable rewiring probability `p`
  - `p = 0` regular lattice
  - `p ~ 0.1` small-world
  - `p = 1` random-like topology
- RK4 numerical integration
- Quantitative synchronization diagnostics:
  - Order parameter `r(t)`
  - Steady-state `<r>` vs coupling `K` (bifurcation / phase transition diagram)
  - Time to synchronization
  - Local phase correlation matrix `C_ij`
  - Correlation vs graph distance
  - Frequency locking (`Omega_i` vs `omega_i`)
  - FFT power spectral density
- Visualization suite:
  - Phase heatmaps and raster plots
  - Network phase maps
  - Phase histograms
  - Topology comparison plots
  - Rewiring impact and finite-size effect plots

## File Structure

- `network.py` : network construction and shortest-path utilities
- `dynamics.py` : Kuramoto RHS + RK4 simulator
- `analysis.py` : synchronization and spectral metrics
- `visualization.py` : plotting functions
- `main.py` : complete experiment runner

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

All figures and interpretation notes are saved in `results/`.
