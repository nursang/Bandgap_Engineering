"""
Trajectory_I1I2_BeamModels_bilayer.py

Compute and plot:
1. The trajectory on the (I1, I2) plane for a 2-segment beam.
2. The dispersion relation of the unit cell.

Supported beam theories:

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

Test-case convention:
    TEST_CASE = 1     -> Euler-Bernoulli test case
    TEST_CASE = 2     -> Rayleigh test case 1
    TEST_CASE = 3     -> Rayleigh test case 2
    TEST_CASE = 4     -> Timoshenko test case 4
    TEST_CASE = None  -> Timoshenko with custom parameters

Dispersion plot convention:
- y-axis: frequency [Hz]
- x-axis: kL
    left side  (x < 0):  imaginary projection, plotted as -|Im(kL)|
    right side (x >= 0): real projection, folded to [0, pi]

Important:
- Bloch branches are computed in a unified way using:
      kL = arccos(y/2)
  with complex arithmetic.
- Continuity is enforced frequency-to-frequency by choosing an equivalent
  representative closest to the previous one.
- Real-part locking to 0 or pi is used only in the natural cases:
    * y real and y >  2  -> Re(kL) = 0
    * y real and y < -2  -> Re(kL) = pi
- In genuinely complex regions (complex y), Re(kL) is NOT locked.
- Same color is used for both branches:
    solid red  -> real projection
    dashed red -> imaginary projection
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
USE_CUSTOM_PARAMETERS = False

# Built-in test case selection:
#   1    -> Euler-Bernoulli test case
#   2    -> Rayleigh test case 1
#   3    -> Rayleigh test case 2
#   4    -> Timoshenko test case 4
#   None -> Timoshenko custom/default setup
TEST_CASE = 4

# If USE_CUSTOM_PARAMETERS = True, choose the beam model manually:
#   1 = Euler-Bernoulli, 2 = Rayleigh, 3 = Timoshenko
CUSTOM_MODEL = 3

# Frequency sampling
FREQ_MIN = 0.1
FREQ_MAX = 1200.0
N_FREQ = 30000

# Plot window in the (I1, I2) plane
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -200.0, 200.0

# Dispersion plot options
PLOT_DISPERSION = True
IMAG_KL_MAX = 4.0

# Numerical tolerances
Y_REAL_TOL = 1e-10
Y_BOUND_TOL = 1e-10
SMALL_IMAG_TOL = 1e-12

# Dispersion styles
DISPERSION_LINEWIDTH = 1.2
DISPERSION_COLOR = "#8B0000"   # dark red
REAL_LINESTYLE = "-"           # solid red for real projection
IMAG_LINESTYLE = "--"          # dashed red for imaginary projection


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
    if USE_CUSTOM_PARAMETERS:
        model = CUSTOM_MODEL

        if model not in MODEL_NAMES:
            raise ValueError("CUSTOM_MODEL must be 1, 2, or 3.")

        params = {
            "L1": 1.35,
            "L2": 3.25,
            #"L2": 0.00,
            "E1": 32e9,
            "rho1": 7850.0,
            "A1": 0.14,
            "I1b": 6.3e-3,
            "kGA1": 5.0e9,
            "E2": 210e9,
            "rho2": 7850.0,
            "A2": 0.018,
            "I2b": 2.9e-3,
            "kGA2": 5.0e9,
            "label": "custom"
        }
        return model, params

    if TEST_CASE == 1:
        model = 1
        params = {
            "L1": 1.35,
            "L2": 3.25,
            "E1": 32e9,
            "rho1": 7850.0,
            "A1": 0.14,
            "I1b": 6.3e-3,
            "kGA1": 1.0e25,
            "E2": 210e9,
            "rho2": 7850.0,
            "A2": 0.018,
            "I2b": 2.9e-3,
            "kGA2": 1.0e25,
            "label": "test_case_1_EB"
        }

    elif TEST_CASE == 2:
        model = 2
        params = {
            "L1": 1.35,
            "L2": 3.25,
            "E1": 32e9,
            "rho1": 785.0 * 1.4,
            "A1": 0.10,
            "I1b": 1.2e-3,
            "kGA1": 1.0e25,
            "E2": 8.53e9,
            "rho2": 7850.0,
            "A2": 1.2e-3,
            "I2b": 1.2e-2,
            "kGA2": 1.0e25,
            "label": "test_case_2_Rayleigh_case1"
        }

    elif TEST_CASE == 3:
        model = 2
        params = {
            "L1": 1.35,
            "L2": 3.25,
            "E1": 32e9,
            "rho1": 785.0 * 0.8,
            "A1": 0.10,
            "I1b": 1.2e-2,
            "kGA1": 1.0e25,
            "E2": 8.53e9,
            "rho2": 7850.0,
            "A2": 1.2e-3,
            "I2b": 1.2e-1,
            "kGA2": 1.0e25,
            "label": "test_case_3_Rayleigh_case2"
        }

    elif TEST_CASE == 4:
        model = 3
        params = {
            "L1": 1.35,
            "L2": 3.25,
            "E1": 32e9,
            "rho1": 785.0 * 0.8,
            "A1": 0.10,
            "I1b": 1.2e-2,
            "kGA1": 5.0e9,
            "E2": 8.53e9,
            "rho2": 7850.0,
            "A2": 1.2e-3,
            "I2b": 1.2e-1,
            "kGA2": 5.0e9,
            "label": "test_case_4_Timoshenko"
        }

    elif TEST_CASE is None:
        model = 3
        params = {
            "L1": 1.35,
            "L2": 3.25,
            "E1": 32e9,
            "rho1": 7850.0,
            "A1": 0.14,
            "I1b": 6.3e-3,
            "kGA1": 5.0e9,
            "E2": 210e9,
            "rho2": 7850.0,
            "A2": 0.018,
            "I2b": 2.9e-3,
            "kGA2": 5.0e9,
            "label": "timoshenko_default"
        }

    else:
        raise ValueError("Invalid TEST_CASE. Use 1, 2, 3, 4, or None.")

    return model, params


MODEL, P = load_configuration()

print(f"Selected beam model: {MODEL_NAMES[MODEL]}")
print(f"Configuration label : {P['label']}")
print()


# =============================================================================
# 3) SYSTEM MATRIX
# =============================================================================
def segment_system_matrix(model, E, I, rho, A, kGA, omega_val):
    EI = E * I
    rhoA = rho * A
    rhoI = rho * I

    if model == 3:
        H = np.array([
            [0.0,                      1.0,                 1.0 / kGA,  0.0],
            [0.0,                      0.0,                 0.0,        1.0 / EI],
            [-(omega_val**2) * rhoA,   0.0,                 0.0,        0.0],
            [0.0, -(omega_val**2) * rhoI,                -1.0,        0.0]
        ], dtype=complex)

    elif model == 2:
        H = np.array([
            [0.0,                      1.0,                 0.0,        0.0],
            [0.0,                      0.0,                 0.0,        1.0 / EI],
            [-(omega_val**2) * rhoA,   0.0,                 0.0,        0.0],
            [0.0, -(omega_val**2) * rhoI,                -1.0,        0.0]
        ], dtype=complex)

    elif model == 1:
        H = np.array([
            [0.0,                      1.0,                 0.0,        0.0],
            [0.0,                      0.0,                 0.0,        1.0 / EI],
            [-(omega_val**2) * rhoA,   0.0,                 0.0,        0.0],
            [0.0,                      0.0,                -1.0,        0.0]
        ], dtype=complex)

    else:
        raise ValueError("Unknown model. MODEL must be 1, 2, or 3.")

    return H


def segment_transfer(model, E, I, rho, A, kGA, Lseg, omega_val):
    H = segment_system_matrix(model, E, I, rho, A, kGA, omega_val)
    eigvals, eigvecs = np.linalg.eig(H)
    exp_diag = np.diag(np.exp(eigvals * Lseg))
    T = eigvecs @ exp_diag @ np.linalg.inv(eigvecs)
    return np.real_if_close(T, tol=1e-10)


# =============================================================================
# 4) INVARIANTS / REGION / UNIFIED BRANCH HELPERS
# =============================================================================
def fold_to_first_brillouin_zone(kL_real):
    """
    Fold real part to [-pi, pi], then plot its magnitude on [0, pi].
    """
    kL_folded = (kL_real + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(kL_folded)


def classify_region(I1, I2):
    """
    Classify the point (I1, I2) into one of the four regions:
        PP, PS, SS, C
    using the same boundaries as the region map.
    """
    I1_abs = abs(I1)
    line_r = 2.0 * I1_abs - 2.0
    line_s = -2.0 * I1_abs - 2.0
    parab = (I1_abs**2) / 4.0 + 2.0

    if I2 > parab:
        return "C"
    elif (I2 <= parab) and (I2 >= line_r) and (I1_abs < 4.0):
        return "PP"
    elif (I2 <= parab) and (I2 >= line_r) and (I1_abs >= 4.0):
        return "SS"
    elif (I2 < line_r) and (I2 >= line_s):
        return "PS"
    else:
        return "SS"


def compute_y1_y2(I1, I2):
    """
    y_{1,2} = 1/2 * (I1 ± sqrt(I1^2 - 4 I2 + 8))
    """
    delta = I1**2 - 4.0 * I2 + 8.0
    root = np.sqrt(delta + 0j)
    y1 = 0.5 * (I1 - root)
    y2 = 0.5 * (I1 + root)
    return y1, y2


def equivalent_k_candidates(k0, n_shift=2):
    """
    Equivalent representatives satisfying cos(kL)=cos(k0):
        kL = ±k0 + 2*pi*m
    """
    cands = []
    for m in range(-n_shift, n_shift + 1):
        cands.append(k0 + 2.0 * np.pi * m)
        cands.append(-k0 + 2.0 * np.pi * m)
    return cands


def choose_kL_branch(y, prev=None, n_shift=2):
    """
    Unified branch selection using complex arccos:
        kL = arccos(y/2)

    Equivalent representatives ±k0 + 2*pi*m are generated and the one
    closest to the previous point is chosen to enforce continuity.

    Convention:
    - prefer representatives with Im(kL) >= 0 when available
      (attenuating convention)
    """
    k0 = np.arccos(y / 2.0 + 0j)
    cands = equivalent_k_candidates(k0, n_shift=n_shift)

    cands_pos_imag = [z for z in cands if np.imag(z) >= -SMALL_IMAG_TOL]
    if cands_pos_imag:
        cands = cands_pos_imag

    if prev is None:
        return min(cands, key=lambda z: (abs(np.imag(z)), abs(np.real(z))))
    return min(cands, key=lambda z: abs(z - prev))


def project_kL_from_y(y, kL):
    """
    Natural projection rule based on y = 2 cos(kL).

    Cases:
    1) y real, |y| <= 2  -> propagating: plot only Re(kL)
    2) y real, y > 2     -> stop branch: Re(kL)=0, plus Im(kL)
    3) y real, y < -2    -> stop branch: Re(kL)=pi, plus Im(kL)
    4) y complex         -> genuinely complex: plot both actual Re(kL), Im(kL)

    This avoids artificial locking in the complex region.
    """
    yr = np.real(y)
    yi = np.imag(y)

    kr = np.real(kL)
    ki = np.imag(kL)

    y_is_real = abs(yi) <= Y_REAL_TOL

    if y_is_real:
        # Propagating branch
        if abs(yr) <= 2.0 + Y_BOUND_TOL:
            x_real = fold_to_first_brillouin_zone(kr)
            x_imag = np.nan
            return x_real, x_imag

        # Stop branch with Re(kL)=0
        if yr > 2.0:
            x_real = 0.0
            x_imag = -abs(ki) if abs(ki) > SMALL_IMAG_TOL else np.nan
            return x_real, x_imag

        # Stop branch with Re(kL)=pi
        x_real = np.pi
        x_imag = -abs(ki) if abs(ki) > SMALL_IMAG_TOL else np.nan
        return x_real, x_imag

    # Genuinely complex branch: do NOT lock
    x_real = fold_to_first_brillouin_zone(kr)
    x_imag = -abs(ki) if abs(ki) > SMALL_IMAG_TOL else np.nan
    return x_real, x_imag


# =============================================================================
# 5) FREQUENCY RANGE
# =============================================================================
f = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
omega = 2.0 * np.pi * f


# =============================================================================
# 6) COMPUTE INVARIANTS I1 AND I2 + DISPERSION DATA
# =============================================================================
I1_vals = []
I2_vals = []

y1_x_real = []
y1_x_imag = []

y2_x_real = []
y2_x_imag = []

region_vals = []

kL1_prev = None
kL2_prev = None

for om in omega:
    M1 = segment_transfer(
        MODEL, P["E1"], P["I1b"], P["rho1"], P["A1"], P["kGA1"], P["L1"], om
    )
    M2 = segment_transfer(
        MODEL, P["E2"], P["I2b"], P["rho2"], P["A2"], P["kGA2"], P["L2"], om
    )

    M = M2 @ M1

    trM = np.trace(M)
    trM2 = np.trace(M @ M)

    I1 = np.real(trM)
    I2 = 0.5 * np.real(trM**2 - trM2)

    I1_vals.append(float(I1))
    I2_vals.append(float(I2))

    region = classify_region(I1, I2)
    region_vals.append(region)

    y1, y2 = compute_y1_y2(I1, I2)

    kL1 = choose_kL_branch(y1, kL1_prev, n_shift=2)
    kL2 = choose_kL_branch(y2, kL2_prev, n_shift=2)

    kL1_prev = kL1
    kL2_prev = kL2

    x1r, x1i = project_kL_from_y(y1, kL1)
    x2r, x2i = project_kL_from_y(y2, kL2)

    y1_x_real.append(float(x1r) if np.isfinite(x1r) else np.nan)
    y1_x_imag.append(float(x1i) if np.isfinite(x1i) else np.nan)

    y2_x_real.append(float(x2r) if np.isfinite(x2r) else np.nan)
    y2_x_imag.append(float(x2i) if np.isfinite(x2i) else np.nan)

I1_vals = np.array(I1_vals)
I2_vals = np.array(I2_vals)
region_vals = np.array(region_vals, dtype=object)

y1_x_real = np.array(y1_x_real, dtype=float)
y1_x_imag = np.array(y1_x_imag, dtype=float)
y2_x_real = np.array(y2_x_real, dtype=float)
y2_x_imag = np.array(y2_x_imag, dtype=float)


# =============================================================================
# 7) BUILD REGION MAP IN THE (I1, I2) PLANE
# =============================================================================
nx, ny = 600, 600
xx = np.linspace(X_MIN, X_MAX, nx)
yy = np.linspace(Y_MIN, Y_MAX, ny)
XX, YY = np.meshgrid(xx, yy)

I1_abs = np.abs(XX)
line_r = 2.0 * I1_abs - 2.0
line_s = -2.0 * I1_abs - 2.0
parab = (I1_abs**2) / 4.0 + 2.0

zone = np.full_like(XX, 1, dtype=int)
zone[YY > parab] = 3
zone[(YY <= parab) & (YY >= line_r) & (I1_abs < 4.0)] = 0
zone[(YY <= parab) & (YY >= line_r) & (I1_abs >= 4.0)] = 2
zone[(YY < line_r) & (YY >= line_s)] = 1
zone[YY < line_s] = 2

REGION_COLORS = {
    "PP": "#fb8c00",
    "PS": "#4caf50",
    "SS": "#1e88e5",
    "C":  "#8e24aa"
}

cmap = ListedColormap([
    REGION_COLORS["PP"],
    REGION_COLORS["PS"],
    REGION_COLORS["SS"],
    REGION_COLORS["C"]
])
norm_regions = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


# =============================================================================
# 8) BUILD TRAJECTORY COLORED BY FREQUENCY
# =============================================================================
points = np.array([I1_vals, I2_vals]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm_line = plt.Normalize(f.min(), f.max())
lc = LineCollection(segments, cmap="Reds_r", norm=norm_line)
lc.set_array(f)
lc.set_linewidth(2.0)


# =============================================================================
# 9) PLOT TRAJECTORY ON (I1, I2) PLANE
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)

ax1.imshow(
    zone,
    origin="lower",
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    cmap=cmap,
    norm=norm_regions,
    aspect="auto",
    alpha=0.6
)

ax1.add_collection(lc)

cbar1 = fig1.colorbar(lc, ax=ax1)
cbar1.set_label("Frequency [Hz]", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

x_line = np.linspace(X_MIN, X_MAX, 1000)
ax1.plot(x_line, 2.0 * np.abs(x_line) - 2.0, color="black", lw=1.0)
ax1.plot(x_line, -2.0 * np.abs(x_line) - 2.0, color="black", lw=1.0)
ax1.plot(x_line, (x_line**2) / 4.0 + 2.0, color="black", lw=1.0)

ax1.scatter(4.0, 6.0, color="black", s=10, zorder=5)

legend_elements = [
    Patch(facecolor=REGION_COLORS["C"], label=r"$C$  – Complex"),
    Patch(facecolor=REGION_COLORS["PP"], label=r"$PP$ – Pass–Pass"),
    Patch(facecolor=REGION_COLORS["PS"], label=r"$PS$ – Pass–Stop"),
    Patch(facecolor=REGION_COLORS["SS"], label=r"$SS$ – Stop–Stop")
]
ax1.legend(handles=legend_elements, loc="lower left", frameon=True, title="Regions")

title_text = MODEL_NAMES[MODEL]
if not USE_CUSTOM_PARAMETERS:
    if TEST_CASE == 1:
        title_text += " (test case 1)"
    elif TEST_CASE == 2:
        title_text += " (test case 2)"
    elif TEST_CASE == 3:
        title_text += " (test case 3)"
    elif TEST_CASE == 4:
        title_text += " (test case 4)"
else:
    title_text += " (custom parameters)"

ax1.set_xlabel(r"$I_1$", fontsize=14)
ax1.set_ylabel(r"$I_2$", fontsize=14)
ax1.set_title(
    rf"Trajectory on the $I_1$–$I_2$ plane, {title_text}",
    fontsize=14,
    pad=20
)

ax1.grid(True, alpha=0.35)
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylim(Y_MIN, Y_MAX)

fig1.tight_layout()


# =============================================================================
# 10) SAVE TRAJECTORY FIGURE
# =============================================================================
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

if USE_CUSTOM_PARAMETERS:
    outfile_name_1 = (
        f"Trajectory_I1I2_{MODEL_NAMES[MODEL].replace('-', '').replace(' ', '')}"
        f"_custom_bilayer.png"
    )
else:
    if TEST_CASE == 1:
        outfile_name_1 = "Trajectory_I1I2_EulerBernoulli_testcase1_bilayer.png"
    elif TEST_CASE == 2:
        outfile_name_1 = "Trajectory_I1I2_Rayleigh_testcase2_bilayer.png"
    elif TEST_CASE == 3:
        outfile_name_1 = "Trajectory_I1I2_Rayleigh_testcase3_bilayer.png"
    elif TEST_CASE == 4:
        outfile_name_1 = "Trajectory_I1I2_Timoshenko_testcase4_bilayer.png"
    else:
        outfile_name_1 = "Trajectory_I1I2_Timoshenko_bilayer.png"

outfile_path_1 = script_dir / outfile_name_1
fig1.savefig(outfile_path_1, dpi=300, bbox_inches="tight")
print(f"Trajectory figure saved to: {outfile_path_1}")


# =============================================================================
# 11) DISPERSION RELATION
# =============================================================================
def plot_projection(ax, freq_vals, x_vals, linestyle, color, linewidth):
    """
    Plot only contiguous finite blocks.
    Points separated by NaNs are not connected.
    """
    n = len(x_vals)
    i = 0

    while i < n:
        while i < n and not np.isfinite(x_vals[i]):
            i += 1
        if i >= n:
            break

        i0 = i
        while i < n and np.isfinite(x_vals[i]):
            i += 1
        i1 = i

        if i1 - i0 >= 2:
            ax.plot(
                x_vals[i0:i1],
                freq_vals[i0:i1],
                linestyle=linestyle,
                color=color,
                lw=linewidth,
                zorder=3
            )


def add_region_bands(ax, freq_vals, region_vals, alpha=0.12):
    """
    Draw horizontal region bands as a single raster image so that
    transitions are sharp and there are no blended patch boundaries.
    """
    region_to_int = {"PP": 0, "PS": 1, "SS": 2, "C": 3}
    vals = np.array([region_to_int[r] for r in region_vals], dtype=int)

    img = np.tile(vals[:, None], (1, 2))

    cmap_bands = ListedColormap([
        REGION_COLORS["PP"],
        REGION_COLORS["PS"],
        REGION_COLORS["SS"],
        REGION_COLORS["C"],
    ])
    norm_bands = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_bands.N)

    df = freq_vals[1] - freq_vals[0] if len(freq_vals) > 1 else 1.0
    y0 = freq_vals[0] - 0.5 * df
    y1 = freq_vals[-1] + 0.5 * df

    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[-IMAG_KL_MAX, np.pi, y0, y1],
        cmap=cmap_bands,
        norm=norm_bands,
        interpolation="nearest",
        alpha=alpha,
        zorder=0
    )


def setup_dispersion_axis(ax):
    ax.axvline(0.0, color="black", lw=1.2, zorder=2)

    left_tick_values = np.array([
        IMAG_KL_MAX,
        0.75 * IMAG_KL_MAX,
        0.50 * IMAG_KL_MAX,
        0.25 * IMAG_KL_MAX
    ])
    left_ticks = -left_tick_values
    left_labels = [f"{val:.2f}" for val in left_tick_values]

    xticks = list(left_ticks) + [0.0, np.pi / 2.0, np.pi]
    xlabels = left_labels + [r"$0$", r"$\pi/2$", r"$\pi$"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)

    ax.set_xlim(-IMAG_KL_MAX, np.pi)
    ax.set_ylim(FREQ_MIN, FREQ_MAX)

    ax.grid(False)
    ax.xaxis.grid(True, alpha=0.35, zorder=1)

    ax.set_ylabel("Frequency [Hz]", fontsize=13)


if PLOT_DISPERSION:
    fig2, (ax2_top, ax2_bot) = plt.subplots(
        2, 1, figsize=(8, 10), dpi=300, sharex=True
    )

    add_region_bands(ax2_top, f, region_vals, alpha=0.12)
    add_region_bands(ax2_bot, f, region_vals, alpha=0.12)

    # -------------------------------------------------------------------------
    # top subplot: branch y1 only
    # -------------------------------------------------------------------------
    plot_projection(
        ax2_top, f, y1_x_real,
        REAL_LINESTYLE, DISPERSION_COLOR,
        DISPERSION_LINEWIDTH
    )
    plot_projection(
        ax2_top, f, y1_x_imag,
        IMAG_LINESTYLE, DISPERSION_COLOR,
        DISPERSION_LINEWIDTH
    )
    setup_dispersion_axis(ax2_top)

    ax2_top.legend(
        handles=[
            plt.Line2D(
                [0], [0],
                color=DISPERSION_COLOR,
                lw=DISPERSION_LINEWIDTH + 0.8,
                linestyle=REAL_LINESTYLE,
                label=r"branch $y_1$: real"
            ),
            plt.Line2D(
                [0], [0],
                color=DISPERSION_COLOR,
                lw=DISPERSION_LINEWIDTH + 0.8,
                linestyle=IMAG_LINESTYLE,
                label=r"branch $y_1$: imaginary"
            ),
        ],
        loc="upper left",
        frameon=True
    )
    ax2_top.set_title(r"Dispersion relation — branch $y_1$", fontsize=14, pad=12)

    # -------------------------------------------------------------------------
    # bottom subplot: branch y2 only
    # -------------------------------------------------------------------------
    plot_projection(
        ax2_bot, f, y2_x_real,
        REAL_LINESTYLE, DISPERSION_COLOR,
        DISPERSION_LINEWIDTH
    )
    plot_projection(
        ax2_bot, f, y2_x_imag,
        IMAG_LINESTYLE, DISPERSION_COLOR,
        DISPERSION_LINEWIDTH
    )
    setup_dispersion_axis(ax2_bot)

    ax2_bot.legend(
        handles=[
            plt.Line2D(
                [0], [0],
                color=DISPERSION_COLOR,
                lw=DISPERSION_LINEWIDTH + 0.8,
                linestyle=REAL_LINESTYLE,
                label=r"branch $y_2$: real"
            ),
            plt.Line2D(
                [0], [0],
                color=DISPERSION_COLOR,
                lw=DISPERSION_LINEWIDTH + 0.8,
                linestyle=IMAG_LINESTYLE,
                label=r"branch $y_2$: imaginary"
            ),
        ],
        loc="upper left",
        frameon=True
    )
    ax2_bot.set_title(r"Dispersion relation — branch $y_2$", fontsize=14, pad=12)
    ax2_bot.set_xlabel(
        r"$kL$  (left: Im$(kL)$, right: Re$(kL)$)",
        fontsize=13
    )

    if USE_CUSTOM_PARAMETERS:
        fig2.suptitle(
            f"Dispersion relation, {MODEL_NAMES[MODEL]} (custom parameters)",
            fontsize=15, y=0.98
        )
    else:
        if TEST_CASE == 1:
            disp_title = "Dispersion relation, Euler-Bernoulli (test case 1)"
        elif TEST_CASE == 2:
            disp_title = "Dispersion relation, Rayleigh (test case 2)"
        elif TEST_CASE == 3:
            disp_title = "Dispersion relation, Rayleigh (test case 3)"
        elif TEST_CASE == 4:
            disp_title = "Dispersion relation, Timoshenko (test case 4)"
        else:
            disp_title = "Dispersion relation, Timoshenko"

        fig2.suptitle(disp_title, fontsize=15, y=0.98)

    fig2.tight_layout(rect=[0, 0, 1, 0.965])

    if USE_CUSTOM_PARAMETERS:
        outfile_name_2 = (
            f"Dispersion_split_bands_{MODEL_NAMES[MODEL].replace('-', '').replace(' ', '')}"
            f"_custom_bilayer.png"
        )
    else:
        if TEST_CASE == 1:
            outfile_name_2 = "Dispersion_split_bands_EulerBernoulli_testcase1_bilayer.png"
        elif TEST_CASE == 2:
            outfile_name_2 = "Dispersion_split_bands_Rayleigh_testcase2_bilayer.png"
        elif TEST_CASE == 3:
            outfile_name_2 = "Dispersion_split_bands_Rayleigh_testcase3_bilayer.png"
        elif TEST_CASE == 4:
            outfile_name_2 = "Dispersion_split_bands_Timoshenko_testcase4_bilayer.png"
        else:
            outfile_name_2 = "Dispersion_split_bands_Timoshenko_bilayer.png"

    outfile_path_2 = script_dir / outfile_name_2
    fig2.savefig(outfile_path_2, dpi=300, bbox_inches="tight")
    print(f"Dispersion figure saved to: {outfile_path_2}")

plt.show()