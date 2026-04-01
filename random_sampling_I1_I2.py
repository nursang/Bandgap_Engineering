import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path


# ============================================================
# Unified random-sampling workflow for 6 beam-model cases.
#
# This script evaluates the invariants (I1, I2) of transfer
# matrices generated from randomly sampled beam parameters for:
#   - Euler-Bernoulli beams
#   - Rayleigh beams
#   - Timoshenko beams
#
# It supports both 1-layer and 2-layer configurations.
#
# Additional randomized parameters:
#   - omega (w): log-uniform in [1, 100]
#   - rho      : log-uniform in [0.1, 10]
# ============================================================


# -----------------------------
# Global controls
# -----------------------------
K = 1.0          # Shear correction factor used in the Timoshenko model
SEED = 0         # Fixed random seed for reproducibility

# Save output figures in the same directory as this script
OUTPUT_DIR = Path(__file__).parent


# -----------------------------
# Case configuration
# -----------------------------
# Each case defines:
#   - output file name
#   - plot title
#   - beam model type
#   - number of layers
#   - number of random samples
#   - plotting limits for (I1, I2)
CASE_CONFIG = {
    1: {
        "name": "eb_1layer",
        "title": "Random Sampling (Euler–Bernoulli, 1 layer)",
        "model": "eb",
        "layers": 1,
        "N": 20000,
        "xlim": (-2000, 2000),
        "ylim": (-4000, 4000)
    },
    2: {
        "name": "eb_2layers",
        "title": "Random Sampling (Euler–Bernoulli, 2 layers)",
        "model": "eb",
        "layers": 2,
        "N": 20000,
        "xlim": (-2000, 2000),
        "ylim": (-4000, 4000)
    },
    3: {
        "name": "rayleigh_1layer",
        "title": "Random Sampling (Rayleigh, 1 layer)",
        "model": "rayleigh",
        "layers": 1,
        "N": 20000,
        "xlim": (-20, 20),
        "ylim": (-40, 40)
    },
    4: {
        "name": "rayleigh_2layers",
        "title": "Random Sampling (Rayleigh, 2 layers)",
        "model": "rayleigh",
        "layers": 2,
        "N": 20000,
        "xlim": (-20, 20),
        "ylim": (-40, 40)
    },
    5: {
        "name": "timo_1layer",
        "title": "Random Sampling (Timoshenko, 1 layer)",
        "model": "timoshenko",
        "layers": 1,
        "N": 20000,
        "xlim": (-20, 20),
        "ylim": (-40, 40)
    },
    6: {
        "name": "timo_2layers",
        "title": "Random Sampling (Timoshenko, 2 layers)",
        "model": "timoshenko",
        "layers": 2,
        "N": 20000,
        "xlim": (-20, 20),
        "ylim": (-40, 40)
    },
}


# -----------------------------
# Utility functions
# -----------------------------
def log_uniform(rng, low, high, size):
    """
    Draw samples from a log-uniform distribution.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    low : float
        Lower bound of the sampling interval (must be positive).
    high : float
        Upper bound of the sampling interval (must be positive).
    size : int
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        Array of log-uniformly distributed samples.
    """
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def build_region_map(x_min, x_max, y_min, y_max, nx=500, ny=500):
    """
    Build a discrete region map over the (I1, I2) plane.

    The regions are defined by the following boundaries:
      - y =  2|x| - 2
      - y = -2|x| - 2
      - y = x^2 / 4 + 2

    Parameters
    ----------
    x_min, x_max : float
        Horizontal plotting bounds.
    y_min, y_max : float
        Vertical plotting bounds.
    nx, ny : int, optional
        Resolution of the background grid.

    Returns
    -------
    numpy.ndarray
        Integer-valued region map for visualization.
    """
    xx = np.linspace(x_min, x_max, nx)
    yy = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xx, yy)

    # Region boundaries expressed in terms of |I1|
    I1_abs = np.abs(XX)
    line_r = 2 * I1_abs - 2
    line_s = -2 * I1_abs - 2
    parab = (I1_abs**2) / 4 + 2

    # Default region assignment
    zone = np.full_like(XX, 1)

    # Assign regions according to the geometric partition
    zone[YY > parab] = 3
    zone[(YY <= parab) & (YY >= line_r) & (I1_abs < 4)] = 0
    zone[(YY <= parab) & (YY >= line_r) & (I1_abs >= 4)] = 2
    zone[(YY < line_r) & (YY >= line_s)] = 1
    zone[YY < line_s] = 2

    return zone


# -----------------------------
# State-space beam models
# -----------------------------
def H_EB(A, E, I, rho, w):
    """
    Construct the state matrix for the Euler-Bernoulli beam model.

    Parameters
    ----------
    A : float
        Cross-sectional area.
    E : float
        Young's modulus.
    I : float
        Second moment of area.
    rho : float
        Mass density.
    w : float
        Angular frequency.

    Returns
    -------
    numpy.ndarray
        4x4 system matrix.
    """
    return np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1 / (E * I)],
        [-rho * w**2 * A, 0, 0, 0],
        [0, 0, -1, 0]
    ])


def H_rayleigh(A, E, I, rho, w):
    """
    Construct the state matrix for the Rayleigh beam model.

    Compared with Euler-Bernoulli, this model includes rotary inertia.

    Parameters
    ----------
    A : float
        Cross-sectional area.
    E : float
        Young's modulus.
    I : float
        Second moment of area.
    rho : float
        Mass density.
    w : float
        Angular frequency.

    Returns
    -------
    numpy.ndarray
        4x4 system matrix.
    """
    return np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1 / (E * I)],
        [-rho * w**2 * A, 0, 0, 0],
        [0, -rho * w**2 * I, -1, 0]
    ])


def H_timo(k, G, A, E, I, rho, w):
    """
    Construct the state matrix for the Timoshenko beam model.

    This model includes both shear deformation and rotary inertia.

    Parameters
    ----------
    k : float
        Shear correction factor.
    G : float
        Shear modulus.
    A : float
        Cross-sectional area.
    E : float
        Young's modulus.
    I : float
        Second moment of area.
    rho : float
        Mass density.
    w : float
        Angular frequency.

    Returns
    -------
    numpy.ndarray
        4x4 system matrix.
    """
    return np.array([
        [0, 1, 1 / (k * G * A), 0],
        [0, 0, 0, 1 / (E * I)],
        [-rho * w**2 * A, 0, 0, 0],
        [0, -rho * w**2 * I, -1, 0]
    ])


# -----------------------------
# Core computation
# -----------------------------
def compute_I(model, segs, rho, w):
    """
    Compute the invariants (I1, I2) of the total transfer matrix.

    For a multilayer beam, the overall transfer matrix is obtained as the
    ordered product of matrix exponentials exp(L * H) for each segment.

    Parameters
    ----------
    model : str
        Beam model identifier: "eb", "rayleigh", or "timoshenko".
    segs : list of dict
        List of segment parameter dictionaries.
    rho : float
        Mass density.
    w : float
        Angular frequency.

    Returns
    -------
    tuple[float, float]
        The real parts of invariants (I1, I2), where:
          I1 = trace(M)
          I2 = 0.5 * (trace(M)^2 - trace(M^2))
    """
    # Initialize total transfer matrix
    M = np.eye(4)

    # Multiply segment transfer matrices in sequence
    for s in segs:
        if model == "eb":
            H = H_EB(s["A"], s["E"], s["I"], rho, w)
        elif model == "rayleigh":
            H = H_rayleigh(s["A"], s["E"], s["I"], rho, w)
        else:
            H = H_timo(K, s["G"], s["A"], s["E"], s["I"], rho, w)

        M = expm(s["L"] * H) @ M

    # Compute trace-based invariants of the total transfer matrix
    tr = np.trace(M)
    tr2 = np.trace(M @ M)

    return float(np.real(tr)), float(np.real(0.5 * (tr**2 - tr2)))


# -----------------------------
# Case evaluation
# -----------------------------
def evaluate_case(case_id):
    """
    Evaluate one configured beam case by random sampling.

    This function:
      1. Samples rho and omega randomly
      2. Samples segment properties for each layer
      3. Computes (I1, I2) for each realization
      4. Filters invalid and out-of-plot-range points

    Parameters
    ----------
    case_id : int
        Identifier of the case in CASE_CONFIG.

    Returns
    -------
    tuple
        (cfg, I1_visible, I2_visible)
    """
    cfg = CASE_CONFIG[case_id]
    rng = np.random.default_rng(SEED)

    N = cfg["N"]

    # Randomize density and angular frequency using log-uniform sampling
    rho_s = log_uniform(rng, 0.1, 10, N)
    w_s = log_uniform(rng, 1, 100, N)

    # Generate random segment properties for each layer
    segs_all = []
    for _ in range(cfg["layers"]):
        seg = {
            "L": rng.uniform(0.2, 3, N),              # Segment length
            "A": log_uniform(rng, 0.2, 15, N),       # Cross-sectional area
            "I": log_uniform(rng, 0.02, 5, N),       # Second moment of area
            "E": log_uniform(rng, 0.05, 15, N),      # Young's modulus
        }

        # Shear modulus is only required for Timoshenko beams
        if cfg["model"] == "timoshenko":
            seg["G"] = log_uniform(rng, 0.05, 15, N)

        segs_all.append(seg)

    # Allocate output arrays for invariants
    I1 = np.empty(N)
    I2 = np.empty(N)

    # Evaluate each random realization independently
    for i in range(N):
        segs = [{k: v[i] for k, v in s.items()} for s in segs_all]
        I1[i], I2[i] = compute_I(cfg["model"], segs, rho_s[i], w_s[i])

    # Remove non-finite results
    mask = np.isfinite(I1) & np.isfinite(I2)
    I1, I2 = I1[mask], I2[mask]

    # Restrict to visible plotting window
    x_min, x_max = cfg["xlim"]
    y_min, y_max = cfg["ylim"]
    vis = (I1 >= x_min) & (I1 <= x_max) & (I2 >= y_min) & (I2 <= y_max)

    return cfg, I1[vis], I2[vis]


# -----------------------------
# Plotting
# -----------------------------
def plot_case(case_id):
    """
    Generate and save the plot for a single case.

    The plot contains:
      - a colored background region map
      - the analytical boundary curves
      - the sampled (I1, I2) points

    Parameters
    ----------
    case_id : int
        Identifier of the case in CASE_CONFIG.
    """
    cfg, x, y = evaluate_case(case_id)

    x_min, x_max = cfg["xlim"]
    y_min, y_max = cfg["ylim"]

    # Build background classification map
    zone = build_region_map(x_min, x_max, y_min, y_max)

    # Colormap for region visualization
    cmap = ListedColormap(["#fb8c00", "#4caf50", "#1e88e5", "#8e24aa"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    # Draw background regions
    ax.imshow(
        zone,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        norm=norm,
        alpha=0.35,
        aspect="auto"
    )

    # Draw analytical boundary curves
    xline = np.linspace(x_min, x_max, 800)
    ax.plot(xline, 2 * np.abs(xline) - 2, "k")
    ax.plot(xline, -2 * np.abs(xline) - 2, "k")
    ax.plot(xline, (xline**2) / 4 + 2, "k")

    # Overlay sampled invariant pairs
    ax.scatter(x, y, s=2, c="black", alpha=0.6)

    # Axis formatting
    ax.set_title(cfg["title"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$I_1$")
    ax.set_ylabel(r"$I_2$")

    # Save figure to disk
    out = OUTPUT_DIR / f"case_{case_id}_{cfg['name']}.png"
    fig.savefig(out, bbox_inches="tight")
    print("Saved:", out)


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    # Set to "all" to process every configured case,
    # or replace with a specific integer case ID (e.g., 1, 2, ..., 6).
    case_to_run = "all"

    if case_to_run == "all":
        for c in CASE_CONFIG:
            print("Running case", c)
            plot_case(c)
    else:
        plot_case(case_to_run)