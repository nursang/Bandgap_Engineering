
"""
Trajectory_I1I2_BeamModels_bilayer.py

Compute and plot the trajectory on the (I1, I2) plane for a 2-segment beam
using one of the following beam theories:

    MODEL = 1  -> Euler-Bernoulli
    MODEL = 2  -> Rayleigh
    MODEL = 3  -> Timoshenko

State vector:
    y = [w, phi, V, M]^T

with first-order system:
    y' = H y

Timoshenko model:
    H =
    [ 0   1    1/(kGA)      0   ]
    [ 0   0      0        1/(EI)]
    [ -w^2 rhoA  0        0      0 ]
    [ 0  -w^2 rhoI  -1     0   ]

Special limits:
    Rayleigh        = Timoshenko limit for kGA -> infinity
    Euler-Bernoulli = Rayleigh limit for rhoI -> 0

The script supports two modes:
1. Custom parameters for a user-defined 2-layer beam.
2. Built-in test cases.

Requested test-case convention:
    TEST_CASE = 1     -> Euler-Bernoulli test case
    TEST_CASE = 2     -> Rayleigh test case 1
    TEST_CASE = 3     -> Rayleigh test case 2
    TEST_CASE = None  -> Timoshenko with custom parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# =============================================================================
# 0) USER CONFIGURATION
# =============================================================================
# Set USE_CUSTOM_PARAMETERS = True to manually define a 2-layer beam.
# Set USE_CUSTOM_PARAMETERS = False to use one of the predefined test cases.
USE_CUSTOM_PARAMETERS = False

# Built-in test case selection:
#   1    -> Euler-Bernoulli test case
#   2    -> Rayleigh test case 1
#   3    -> Rayleigh test case 2
#   None -> Timoshenko (recommended with custom parameters)
TEST_CASE = 1

# If USE_CUSTOM_PARAMETERS = True, choose the beam model manually:
#   1 = Euler-Bernoulli, 2 = Rayleigh, 3 = Timoshenko
CUSTOM_MODEL = 2

# Frequency sampling
FREQ_MIN = 0.1       # Minimum frequency [Hz]
FREQ_MAX = 1200.0    # Maximum frequency [Hz]
N_FREQ = 30000       # Number of frequency samples [-]

# Plot window in the (I1, I2) plane
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -200.0, 200.0


# =============================================================================
# 1) MODEL DEFINITIONS
# =============================================================================
MODEL_NAMES = {
    1: "Euler-Bernoulli",
    2: "Rayleigh",
    3: "Timoshenko"
}


# =============================================================================
# 2) PARAMETER SELECTION
# =============================================================================
def load_configuration():
    """
    Return the selected model and the corresponding parameters for a 2-segment
    beam.

    Returns
    -------
    model : int
        1 = Euler-Bernoulli, 2 = Rayleigh, 3 = Timoshenko

    params : dict
        Dictionary containing all segment properties:
            L1, L2     : segment lengths [m]
            E1, E2     : Young's moduli [Pa]
            rho1, rho2 : mass densities [kg/m^3]
            A1, A2     : cross-sectional areas [m^2]
            I1b, I2b   : second moments of area [m^4]
            kGA1, kGA2 : shear rigidities [N]
            label      : configuration label [-]
    """
    if USE_CUSTOM_PARAMETERS:
        # ---------------------------------------------------------------------
        # CUSTOM 2-LAYER INPUT
        # ---------------------------------------------------------------------
        model = CUSTOM_MODEL

        if model not in MODEL_NAMES:
            raise ValueError("CUSTOM_MODEL must be 1, 2, or 3.")

        params = {
            # Geometry
            "L1": 1.35,      # Segment 1 length [m]
            "L2": 3.25,      # Segment 2 length [m]

            # Segment 1
            "E1": 32e9,      # Young's modulus [Pa]
            "rho1": 7850.0,  # Mass density [kg/m^3]
            "A1": 0.14,      # Cross-sectional area [m^2]
            "I1b": 6.3e-3,   # Second moment of area [m^4]
            "kGA1": 1.0e25,  # Shear rigidity [N]

            # Segment 2
            "E2": 210e9,     # Young's modulus [Pa]
            "rho2": 7850.0,  # Mass density [kg/m^3]
            "A2": 0.018,     # Cross-sectional area [m^2]
            "I2b": 2.9e-3,   # Second moment of area [m^4]
            "kGA2": 1.0e25,  # Shear rigidity [N]

            "label": "custom"
        }

        return model, params

    # -------------------------------------------------------------------------
    # BUILT-IN TEST CASES
    # -------------------------------------------------------------------------
    if TEST_CASE == 1:
        # Euler-Bernoulli test case
        model = 1
        params = {
            "L1": 1.35,      # Segment 1 length [m]
            "L2": 3.25,      # Segment 2 length [m]

            "E1": 32e9,      # Young's modulus [Pa]
            "rho1": 7850.0,  # Mass density [kg/m^3]
            "A1": 0.14,      # Cross-sectional area [m^2]
            "I1b": 6.3e-3,   # Second moment of area [m^4]
            "kGA1": 1.0e25,  # Shear rigidity [N]

            "E2": 210e9,     # Young's modulus [Pa]
            "rho2": 7850.0,  # Mass density [kg/m^3]
            "A2": 0.018,     # Cross-sectional area [m^2]
            "I2b": 2.9e-3,   # Second moment of area [m^4]
            "kGA2": 1.0e25,  # Shear rigidity [N]

            "label": "test_case_1_EB"
        }

    elif TEST_CASE == 2:
        # Rayleigh test case 1
        model = 2
        params = {
            "L1": 1.35,         # Segment 1 length [m]
            "L2": 3.25,         # Segment 2 length [m]

            "E1": 32e9,         # Young's modulus [Pa]
            "rho1": 785.0 * 1.4,# Mass density [kg/m^3]
            "A1": 0.10,         # Cross-sectional area [m^2]
            "I1b": 1.2e-3,      # Second moment of area [m^4]
            "kGA1": 1.0e25,     # Shear rigidity [N]

            "E2": 8.53e9,       # Young's modulus [Pa]
            "rho2": 7850.0,     # Mass density [kg/m^3]
            "A2": 1.2e-3,       # Cross-sectional area [m^2]
            "I2b": 1.2e-2,      # Second moment of area [m^4]
            "kGA2": 1.0e25,     # Shear rigidity [N]

            "label": "test_case_2_Rayleigh_case1"
        }

    elif TEST_CASE == 3:
        # Rayleigh test case 2
        model = 2
        params = {
            "L1": 1.35,         # Segment 1 length [m]
            "L2": 3.25,         # Segment 2 length [m]

            "E1": 32e9,         # Young's modulus [Pa]
            "rho1": 785.0 * 0.8,# Mass density [kg/m^3]
            "A1": 0.10,         # Cross-sectional area [m^2]
            "I1b": 1.2e-2,      # Second moment of area [m^4]
            "kGA1": 1.0e25,     # Shear rigidity [N]

            "E2": 8.53e9,       # Young's modulus [Pa]
            "rho2": 7850.0,     # Mass density [kg/m^3]
            "A2": 1.2e-3,       # Cross-sectional area [m^2]
            "I2b": 1.2e-1,      # Second moment of area [m^4]
            "kGA2": 1.0e25,     # Shear rigidity [N]

            "label": "test_case_3_Rayleigh_case2"
        }

    elif TEST_CASE is None:
        # Timoshenko should be used without a built-in test case, as requested.
        # A default custom-like setup is provided here for convenience.
        model = 3
        params = {
            "L1": 1.35,      # Segment 1 length [m]
            "L2": 3.25,      # Segment 2 length [m]

            "E1": 32e9,      # Young's modulus [Pa]
            "rho1": 7850.0,  # Mass density [kg/m^3]
            "A1": 0.14,      # Cross-sectional area [m^2]
            "I1b": 6.3e-3,   # Second moment of area [m^4]
            "kGA1": 5.0e9,   # Shear rigidity [N]

            "E2": 210e9,     # Young's modulus [Pa]
            "rho2": 7850.0,  # Mass density [kg/m^3]
            "A2": 0.018,     # Cross-sectional area [m^2]
            "I2b": 2.9e-3,   # Second moment of area [m^4]
            "kGA2": 5.0e9,   # Shear rigidity [N]

            "label": "timoshenko_default"
        }

    else:
        raise ValueError("Invalid TEST_CASE. Use 1, 2, 3, or None.")

    return model, params


MODEL, P = load_configuration()

print(f"Selected beam model: {MODEL_NAMES[MODEL]}")
print(f"Configuration label : {P['label']}")
print()


# =============================================================================
# 3) SYSTEM MATRIX
# =============================================================================
def segment_system_matrix(model, E, I, rho, A, kGA, omega_val):
    """
    Build the first-order system matrix H for a single beam segment.

    Parameters
    ----------
    model : int
        Beam model selector:
            1 = Euler-Bernoulli
            2 = Rayleigh
            3 = Timoshenko

    E : float
        Young's modulus [Pa].

    I : float
        Second moment of area [m^4].

    rho : float
        Mass density [kg/m^3].

    A : float
        Cross-sectional area [m^2].

    kGA : float
        Shear rigidity [N].
        For Rayleigh and Euler-Bernoulli this can be taken very large.

    omega_val : float
        Angular frequency [rad/s].

    Returns
    -------
    H : (4, 4) ndarray of complex
        State-space matrix such that y' = H y.
    """
    EI = E * I      # Bending stiffness [N·m^2]
    rhoA = rho * A  # Mass per unit length [kg/m]
    rhoI = rho * I  # Rotary inertia per unit length [kg·m]

    if model == 3:
        # Timoshenko model
        H = np.array([
            [0.0,                  1.0,                1.0 / kGA,  0.0],
            [0.0,                  0.0,                0.0,        1.0 / EI],
            [-(omega_val**2)*rhoA, 0.0,                0.0,        0.0],
            [0.0, -(omega_val**2)*rhoI,               -1.0,        0.0]
        ], dtype=complex)

    elif model == 2:
        # Rayleigh model: Timoshenko limit for kGA -> infinity
        H = np.array([
            [0.0,                  1.0,                0.0,        0.0],
            [0.0,                  0.0,                0.0,        1.0 / EI],
            [-(omega_val**2)*rhoA, 0.0,                0.0,        0.0],
            [0.0, -(omega_val**2)*rhoI,               -1.0,        0.0]
        ], dtype=complex)

    elif model == 1:
        # Euler-Bernoulli model: Rayleigh limit for rhoI -> 0
        H = np.array([
            [0.0,                  1.0,                0.0,        0.0],
            [0.0,                  0.0,                0.0,        1.0 / EI],
            [-(omega_val**2)*rhoA, 0.0,                0.0,        0.0],
            [0.0,                  0.0,               -1.0,        0.0]
        ], dtype=complex)

    else:
        raise ValueError("Unknown model. MODEL must be 1, 2, or 3.")

    return H


def segment_transfer(model, E, I, rho, A, kGA, Lseg, omega_val):
    """
    Compute the transfer matrix for one beam segment by matrix exponential
    through eigen-decomposition.

    Parameters
    ----------
    model : int
        Beam model selector [-].

    E : float
        Young's modulus [Pa].

    I : float
        Second moment of area [m^4].

    rho : float
        Mass density [kg/m^3].

    A : float
        Cross-sectional area [m^2].

    kGA : float
        Shear rigidity [N].

    Lseg : float
        Segment length [m].

    omega_val : float
        Angular frequency [rad/s].

    Returns
    -------
    T : (4, 4) ndarray
        Transfer matrix of the beam segment.
    """
    H = segment_system_matrix(model, E, I, rho, A, kGA, omega_val)

    eigvals, eigvecs = np.linalg.eig(H)
    exp_diag = np.diag(np.exp(eigvals * Lseg))
    T = eigvecs @ exp_diag @ np.linalg.inv(eigvecs)

    return np.real_if_close(T, tol=1e-10)


# =============================================================================
# 4) FREQUENCY RANGE
# =============================================================================
f = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)  # Frequency [Hz]
omega = 2.0 * np.pi * f                      # Angular frequency [rad/s]


# =============================================================================
# 5) COMPUTE INVARIANTS I1 AND I2
# =============================================================================
I1_vals = []
I2_vals = []

for om in omega:
    M1 = segment_transfer(
        MODEL, P["E1"], P["I1b"], P["rho1"], P["A1"], P["kGA1"], P["L1"], om
    )
    M2 = segment_transfer(
        MODEL, P["E2"], P["I2b"], P["rho2"], P["A2"], P["kGA2"], P["L2"], om
    )

    # Total transfer matrix for the 2-segment beam
    M = M2 @ M1

    trM = np.trace(M)
    trM2 = np.trace(M @ M)

    I1_vals.append(float(np.real(trM)))
    I2_vals.append(float(0.5 * np.real(trM**2 - trM2)))

I1_vals = np.array(I1_vals)
I2_vals = np.array(I2_vals)


# =============================================================================
# 6) BUILD REGION MAP IN THE (I1, I2) PLANE
# =============================================================================
nx, ny = 600, 600
xx = np.linspace(X_MIN, X_MAX, nx)
yy = np.linspace(Y_MIN, Y_MAX, ny)
XX, YY = np.meshgrid(xx, yy)

I1_abs = np.abs(XX)
line_r = 2.0 * I1_abs - 2.0
line_s = -2.0 * I1_abs - 2.0
parab = (I1_abs**2) / 4.0 + 2.0

# Region coding:
#   0 -> PP
#   1 -> PS
#   2 -> SS
#   3 -> C
zone = np.full_like(XX, 1, dtype=int)
zone[YY > parab] = 3
zone[(YY <= parab) & (YY >= line_r) & (I1_abs < 4.0)] = 0
zone[(YY <= parab) & (YY >= line_r) & (I1_abs >= 4.0)] = 2
zone[(YY < line_r) & (YY >= line_s)] = 1
zone[YY < line_s] = 2

cmap = ListedColormap([
    "#fb8c00",  # PP
    "#4caf50",  # PS
    "#1e88e5",  # SS
    "#8e24aa"   # C
])
norm_regions = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


# =============================================================================
# 7) BUILD TRAJECTORY COLORED BY FREQUENCY
# =============================================================================
points = np.array([I1_vals, I2_vals]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm_line = plt.Normalize(f.min(), f.max())
lc = LineCollection(segments, cmap="Reds_r", norm=norm_line)
lc.set_array(f)
lc.set_linewidth(2.0)


# =============================================================================
# 8) PLOT RESULTS
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

# Background region map
ax.imshow(
    zone,
    origin="lower",
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    cmap=cmap,
    norm=norm_regions,
    aspect="auto",
    alpha=0.6
)

# Frequency-colored trajectory
ax.add_collection(lc)

# Colorbar
cbar = fig.colorbar(lc, ax=ax)
cbar.set_label("Frequency [Hz]", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Boundary curves
x_line = np.linspace(X_MIN, X_MAX, 1000)
ax.plot(x_line,  2.0 * np.abs(x_line) - 2.0, color="black", lw=1.0)
ax.plot(x_line, -2.0 * np.abs(x_line) - 2.0, color="black", lw=1.0)
ax.plot(x_line, (x_line**2) / 4.0 + 2.0, color="black", lw=1.0)

# Mark the special point
ax.scatter(4.0, 6.0, color="black", s=10, zorder=5)

# Legend
legend_elements = [
    Patch(facecolor="#8e24aa", label=r"$C$  – Complex"),
    Patch(facecolor="#fb8c00", label=r"$PP$ – Pass–Pass"),
    Patch(facecolor="#4caf50", label=r"$PS$ – Pass–Stop"),
    Patch(facecolor="#1e88e5", label=r"$SS$ – Stop–Stop")
]
ax.legend(handles=legend_elements, loc="lower left", frameon=True, title="Regions")

# Title
title_text = MODEL_NAMES[MODEL]
if not USE_CUSTOM_PARAMETERS:
    if TEST_CASE == 1:
        title_text += " (test case 1)"
    elif TEST_CASE == 2:
        title_text += " (test case 2)"
    elif TEST_CASE == 3:
        title_text += " (test case 3)"
else:
    title_text += " (custom parameters)"

ax.set_xlabel(r"$I_1$", fontsize=14)
ax.set_ylabel(r"$I_2$", fontsize=14)
ax.set_title(
    rf"Trajectory on the $I_1$–$I_2$ plane, {title_text}",
    fontsize=14,
    pad=20
)

ax.grid(True, alpha=0.35)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

plt.tight_layout()


# =============================================================================
# 9) SAVE FIGURE IN THE SCRIPT DIRECTORY
# =============================================================================
if USE_CUSTOM_PARAMETERS:
    outfile_name = (
        f"Trajectory_I1I2_{MODEL_NAMES[MODEL].replace('-', '').replace(' ', '')}"
        f"_custom_bilayer.png"
    )
else:
    if TEST_CASE == 1:
        outfile_name = "Trajectory_I1I2_EulerBernoulli_testcase1_bilayer.png"
    elif TEST_CASE == 2:
        outfile_name = "Trajectory_I1I2_Rayleigh_testcase2_bilayer.png"
    elif TEST_CASE == 3:
        outfile_name = "Trajectory_I1I2_Rayleigh_testcase3_bilayer.png"
    else:
        outfile_name = "Trajectory_I1I2_Timoshenko_bilayer.png"

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

outfile_path = script_dir / outfile_name

plt.savefig(outfile_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {outfile_path}")

plt.show()