#!/usr/bin/env python3
"""
post_check_full6dof_position_estimation.py
-----------------------------------------
Forward-integrates full 6DOF Fossen EOM and evaluates POSITION / ATTITUDE estimation.

Core model:
    M nu_dot + C(nu)nu + D(nu)nu + g(eta) = tau
    eta_dot = J(eta) nu

Preserved features:
    - segmented integration (NaN/dropout aware)
    - validity gating / age gating
    - INCLUDE_CA / INCLUDE_G toggles
    - optional tau scaling
    - optional fixed dt
    - parameter-set toggles (MLE/MAP/SIM_TRUTH/etc)
    - optional HMC posterior predictive band (segmented + filtered)

Removed (per request):
    - accel residual / consistency check
    - tau residual inverse dynamics diagnostics
    - all velocity plotting / DOF selection
"""

import numpy as np
import matplotlib.pyplot as plt
import torch  # only needed if you enable HMC band loading
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

# ============================================================
# === 0) Config / Flags ======================================
# ============================================================

# ------------------------------------------------------------
# 0a) Paths (you set these manually)
# ------------------------------------------------------------
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Coupled Maneuvers/defender_data_teleop_circle.csv"
HMC_SAMPLES_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_surge_samples.pt"

# --- NaN / dropout handling ---
SEGMENTED_INTEGRATION = False         # integrate only over valid segments; NaNs stay as gaps
USE_VALIDITY_FLAGS = True             # gate using pose/twist/wrench valid fields if present
REQUIRE_POSE_VALID = False
REQUIRE_TWIST_VALID = True
REQUIRE_WRENCH_VALID = True
MAX_ALLOWED_AGE_S = None              # e.g. 0.05, or None
MIN_SEG_LEN = 50                      # samples

# ------------------------------------------------------------
# 0b) Physics toggles
# ------------------------------------------------------------
INCLUDE_CA = True   # True -> use C_RB + C_A, False -> use C_RB only
INCLUDE_G  = True   # include restoring g(eta)

# ------------------------------------------------------------
# 0j) Optional thrust scaling (LUT gain test)
# ------------------------------------------------------------
USE_TAU_SCALE = False
TAU_SCALE = 1.1   # try 0.9, 1.1, etc.

# ------------------------------------------------------------
# 0d) Integrator timing
# ------------------------------------------------------------
FORCE_FIXED_DT = True
DT_FIXED = 0.01

# ------------------------------------------------------------
# 0e) Which parameter sets to run
# ------------------------------------------------------------
PLOT_SIM_MLE   = False
PLOT_SIM_TRUTH = False
PLOT_MLE       = True

PLOT_TEST_PARAMS_1 = False
PLOT_TEST_PARAMS_2 = False

PLOT_MAP = False
PLOT_HMC_TEST_PARAMS_1 = False

# ------------------------------------------------------------
# 0f) HMC posterior predictive band (6DOF forward integration)
# ------------------------------------------------------------
ENABLE_HMC_BAND = False
HMC_BAND_NPLOT  = 200

HMC_BASELINE = "SIM_MLE"   # "MAP" | "SIM_TRUTH" | "SIM_MLE"

HMC_BAND_PARAMS_BY_DOF = {
    "X": ["X_dot_u", "X_u", "X_uu"],
    "Y": ["Y_dot_v", "Y_v", "Y_vv"],
    "Z": ["Z_dot_w", "Z_w", "Z_ww"],
    "K": ["K_dot_p", "K_p", "K_pp"],
    "M": ["M_dot_q", "M_q", "M_qq"],
    "N": ["N_dot_r", "N_r", "N_rr"],
}
HMC_BAND_GLOBAL_EXTRAS = [
     "B"
    # "x_cg", "y_cg", "z_cg",
    # "x_cb", "y_cb", "z_cb",
]
HMC_BAND_DRIVER = "N"  # choose which DOF subset list to sample for band
HMC_BAND_PARAM_SUBSET = HMC_BAND_PARAMS_BY_DOF[HMC_BAND_DRIVER] + HMC_BAND_GLOBAL_EXTRAS

# ------------------------------------------------------------
# 0k) Position outputs to evaluate/plot
# eta indices: [x, y, z, phi, theta, psi]
# ------------------------------------------------------------
EVAL_COMPONENTS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "psi": 5,   # yaw
}
ANGLE_COMPONENTS = {"phi", "theta", "psi"}

# Plot window (optional)
T_WINDOW = None  # seconds, e.g. 30.0, or None for full

# NEW: start time for plotting/metrics (relative time in seconds)
T_START = None   # e.g. 63.0, or None for full (65 is past the turnaround)

# ------------------------------------------------------------
# 0i) quick config printouts
# ------------------------------------------------------------
print(f"[config] INCLUDE_CA={INCLUDE_CA}, INCLUDE_G={INCLUDE_G}")
print(f"[config] SEGMENTED_INTEGRATION={SEGMENTED_INTEGRATION}, USE_VALIDITY_FLAGS={USE_VALIDITY_FLAGS}")
print(f"[config] HMC_BASELINE={HMC_BASELINE}, ENABLE_HMC_BAND={ENABLE_HMC_BAND}")
print(f"[config] HMC_BAND_PARAM_SUBSET={HMC_BAND_PARAM_SUBSET}")
print(f"[config] EVAL_COMPONENTS={EVAL_COMPONENTS}, T_WINDOW={T_WINDOW}")

# ============================================================
# === 1) Load CSV ============================================
# ============================================================
data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)
if data.ndim != 2 or data.shape[1] < 25:
    raise RuntimeError(f"CSV seems malformed or too few columns: shape={data.shape}")

t = data[:, 0].astype(float)
t_rel = t - t[0]

# ============================================================
# === Optional: trim data to start at T_START =================
# ============================================================
if T_START is not None:
    T_START = float(T_START)
    # first index where t_rel >= T_START
    i_start = int(np.searchsorted(t_rel, T_START, side="left"))
    i_start = max(0, min(i_start, len(t_rel) - 1))

    print(f"[INFO] Trimming run to start at T_START={T_START:.3f}s (index {i_start}, t_rel={t_rel[i_start]:.3f}s)")

    # Slice raw time first
    t = t[i_start:]
    t_rel = t_rel[i_start:] - t_rel[i_start]   # re-zero time at the new start

    # IMPORTANT: everything else is sliced later *after* you build it
    _i_start = i_start
else:
    _i_start = 0


# ν_dot (1..6) — loaded but not used (kept for compatibility with CSV format)
nu_dot_meas = np.vstack([data[:, i].astype(float) for i in range(1, 7)]).T

# ν (7..12)
nu_meas = np.vstack([data[:, i].astype(float) for i in range(7, 13)]).T

# η (13..18)
eta_meas = np.vstack([data[:, i].astype(float) for i in range(13, 19)]).T

# τ (19..24)
tau = np.vstack([data[:, i].astype(float) for i in range(19, 25)]).T

# Optional validity columns (25..30) if present
pose_valid = pose_age = twist_valid = twist_age = wrench_valid = wrench_age = None
if data.shape[1] >= 31:
    pose_valid   = data[:, 25].astype(float)
    pose_age     = data[:, 26].astype(float)
    twist_valid  = data[:, 27].astype(float)
    twist_age    = data[:, 28].astype(float)
    wrench_valid = data[:, 29].astype(float)
    wrench_age   = data[:, 30].astype(float)


# Apply the same start trim to all signals (if enabled)
if _i_start > 0:
    nu_dot_meas = nu_dot_meas[_i_start:, :]
    nu_meas     = nu_meas[_i_start:, :]
    eta_meas    = eta_meas[_i_start:, :]
    tau         = tau[_i_start:, :]

    # trim validity vectors too (if present)
    if pose_valid is not None:
        pose_valid   = pose_valid[_i_start:]
        pose_age     = pose_age[_i_start:]
        twist_valid  = twist_valid[_i_start:]
        twist_age    = twist_age[_i_start:]
        wrench_valid = wrench_valid[_i_start:]
        wrench_age   = wrench_age[_i_start:]


if USE_TAU_SCALE:
    print(f"[INFO] Scaling tau by factor {TAU_SCALE}")
    tau = TAU_SCALE * tau


def pct_valid(x):
    if x is None:
        return None
    return 100.0 * float(np.mean(x > 0.5))

print("\n=== CSV columns ===")
print(f"shape = {data.shape}")
if data.shape[1] >= 31:
    print(f"pose_valid  : {pct_valid(pose_valid):.2f}% valid")
    print(f"twist_valid : {pct_valid(twist_valid):.2f}% valid")
    print(f"wrench_valid: {pct_valid(wrench_valid):.2f}% valid")
else:
    print("No pose/twist/wrench validity columns detected (need 31 cols).")
print("===================\n")

# Optional time window mask (used for plotting/metrics)
time_window_mask = np.ones_like(t_rel, dtype=bool)
if T_WINDOW is not None:
    time_window_mask &= (t_rel <= float(T_WINDOW))



# ============================================================
# === 1b) Segments (NaN/dropout aware) =======================
# ============================================================
def contiguous_segments(mask: np.ndarray, min_len: int = 10):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, idx.size - 1]
    segs_out = []
    for s, e in zip(starts, ends):
        a = int(idx[s])
        b = int(idx[e] + 1)
        if (b - a) >= min_len:
            segs_out.append((a, b))
    return segs_out

def build_valid_mask(
    nu_meas: np.ndarray,
    eta_meas: np.ndarray,
    tau: np.ndarray,
    pose_valid=None, pose_age=None,
    twist_valid=None, twist_age=None,
    wrench_valid=None, wrench_age=None,
    use_flags: bool = True,
    require_pose: bool = False,
    require_twist: bool = True,
    require_wrench: bool = True,
    max_age_s=None,
):
    m = (
        np.isfinite(nu_meas).all(axis=1) &
        np.isfinite(eta_meas).all(axis=1) &
        np.isfinite(tau).all(axis=1)
    )

    if use_flags and (pose_valid is not None) and (twist_valid is not None) and (wrench_valid is not None):
        if require_pose:
            m &= (pose_valid > 0.5)
        if require_twist:
            m &= (twist_valid > 0.5)
        if require_wrench:
            m &= (wrench_valid > 0.5)

        if max_age_s is not None:
            if require_pose and (pose_age is not None):
                m &= (pose_age <= max_age_s)
            if require_twist and (twist_age is not None):
                m &= (twist_age <= max_age_s)
            if require_wrench and (wrench_age is not None):
                m &= (wrench_age <= max_age_s)

    return m

valid_mask = build_valid_mask(
    nu_meas=nu_meas,
    eta_meas=eta_meas,
    tau=tau,
    pose_valid=pose_valid, pose_age=pose_age,
    twist_valid=twist_valid, twist_age=twist_age,
    wrench_valid=wrench_valid, wrench_age=wrench_age,
    use_flags=USE_VALIDITY_FLAGS,
    require_pose=REQUIRE_POSE_VALID,
    require_twist=REQUIRE_TWIST_VALID,
    require_wrench=REQUIRE_WRENCH_VALID,
    max_age_s=MAX_ALLOWED_AGE_S,
)

segs = contiguous_segments(valid_mask, min_len=MIN_SEG_LEN) if SEGMENTED_INTEGRATION else [(0, len(t))]

total_valid = int(np.sum(valid_mask))
print(f"[INFO] valid_mask True samples: {total_valid} / {len(valid_mask)} ({100*total_valid/len(valid_mask):.2f}%)")
print(f"[INFO] Found {len(segs)} segment(s) (min_len={MIN_SEG_LEN})")
if len(segs) == 0:
    raise RuntimeError("No valid segments found. Relax validity gating or check CSV.")

# ============================================================
# === 2) Parameters ==========================================
# ============================================================
G = 9.8
params_base = {"W": 23.89 * G}

params_mle = {
    **params_base,
    "m": 23.89, "B": 236.00, "I_xx": 0.5, "I_yy": 1.76, "I_zz": 2.13,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,
    "X_dot_u": -33.61, "Y_dot_v": -31.56, "Z_dot_w": -79.58, "K_dot_p": -0.1, "M_dot_q": -0.46, "N_dot_r": -0.70,
    "X_u": -16.49, "X_uu": -42.49,
    "Y_v": -34.05, "Y_vv": -108.74,
    "Z_w": -35.66, "Z_ww": -128.31,
    "K_p": -1.24, "K_pp": -0.08,
    "M_q": -2.08, "M_qq": -1.61,
    "N_r": -2.88, "N_rr": -2.69,
}

params_sim_truth = {
    **params_base,
    "m": 17.2, "B": 17.2 * G, "I_xx": 1.0, "I_yy": 1.0, "I_zz": 1.0,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,
    "X_dot_u": -18.0, "Y_dot_v": -22.584, "Z_dot_w": -22.3775, "K_dot_p": -0.079, "M_dot_q": -0.26, "N_dot_r": -0.286,
    "X_u": -4.66, "X_uu": -51.5,
    "Y_v": -8.25, "Y_vv": -102.006,
    "Z_w": -14.17, "Z_ww": -155.8358,
    "K_p": -1.5, "K_pp": -2.1,
    "M_q": -2.9, "M_qq": -14.6,
    "N_r": -10.343, "N_rr": -8.8,
}

params_sim_mle = {
    **params_base,
    "m": 17.2, "B": 168.56, "I_xx": 1.0, "I_yy": 1.0, "I_zz": 1.0,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,
    "X_dot_u": -17.44, "Y_dot_v": -22.12, "Z_dot_w": -21.60, "K_dot_p": -0.07, "M_dot_q": -0.24, "N_dot_r": -0.26,
    "X_u": -4.72, "X_uu": -51.4,
    "Y_v": -8.31, "Y_vv": -101.88,
    "Z_w": -14.23, "Z_ww": -155.57,
    "K_p": -1.44, "K_pp": -2.08,
    "M_q": -2.9, "M_qq": -14.36,
    "N_r": -10.37, "N_rr": -8.75,
}

params_test_1 = {
    **params_base,
    "m": 23.8, "B": 239.80, "I_xx": 1.0, "I_yy": 1.0, "I_zz": 1.442503809928894,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.02,
    "X_dot_u": -18.754636764526367, "Y_dot_v": -19.286680221557617, "Z_dot_w": -54.19, "K_dot_p": -0.079, "M_dot_q": -0.26, "N_dot_r": -1.4425019025802612,
    "X_u": -15.252839088439941, "X_uu": -43.881591796875,
    "Y_v": -38.239105224609375, "Y_vv": -117.48196411132812,
    "Z_w": -33.47, "Z_ww": -136.92,
    "K_p": -1.5, "K_pp": -2.1,
    "M_q": -2.9, "M_qq": -14.6,
    "N_r": -2.8859241008758545, "N_rr": -2.6936986446380615,
}

params_test_2 = {
    **params_base,
    "m": 23.89, "B": 235.89, "I_xx": 0.5, "I_yy": 1.76, "I_zz": 2.13,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,
    "X_dot_u": -33.61, "Y_dot_v": -11.48, "Z_dot_w": -56.142, "K_dot_p": -0.1, "M_dot_q": -0.46, "N_dot_r": -0.70,
    "X_u": -16.49, "X_uu": -42.49,
    "Y_v": -27.27, "Y_vv": -107.24,
    "Z_w": -33.38, "Z_ww": -137.14,
    "K_p": -1.24, "K_pp": -0.08,
    "M_q": -2.08, "M_qq": -1.61,
    "N_r": -2.88, "N_rr": -2.69,
}

params_map = {
    **params_base,
    "m": 23.89, "B": 235.97, "I_xx": 0.41, "I_yy": 1.31, "I_zz": 1.46,
    "x_cg": -0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,
    "X_dot_u": -32.41, "Y_dot_v": -16.78, "Z_dot_w": -77.77, "K_dot_p": -0.22, "M_dot_q": -0.91, "N_dot_r": -1.32,
    "X_u": -1.01, "X_uu": -62.05,
    "Y_v": -0.93, "Y_vv": -137.19,
    "Z_w": -35.30, "Z_ww": -126.63,
    "K_p": -1.14, "K_pp": -0.2,
    "M_q": -1.05, "M_qq": -2.91,
    "N_r": -0.99, "N_rr": -3.51,
}

params_hmc_test_1 = dict(params_map)

PARAM_SETS = {
    "MLE": params_mle,
    "TEST_1": params_test_1,
    "TEST_2": params_test_2,
    "SIM_TRUTH": params_sim_truth,
    "SIM_MLE": params_sim_mle,
    "MAP": params_map,
    "HMC_TEST_1": params_hmc_test_1,
}

# ============================================================
# === 2b) Optional: load HMC samples for predictive band ======
# ============================================================
theta_plot = None
param_names = None
if ENABLE_HMC_BAND:
    hmc_data = torch.load(HMC_SAMPLES_PATH, map_location="cpu")
    theta_samples = hmc_data["samples"].detach().cpu().numpy()
    param_names = list(hmc_data["param_names"])
    if theta_samples.shape[0] > HMC_BAND_NPLOT:
        sel = np.random.choice(theta_samples.shape[0], size=HMC_BAND_NPLOT, replace=False)
        theta_plot = theta_samples[sel]
    else:
        theta_plot = theta_samples

# ============================================================
# === 3) Helpers (Full 6DOF Fossen) ==========================
# ============================================================
def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2*np.pi) - np.pi

def skew(a: np.ndarray) -> np.ndarray:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    return np.array([[0.0, -az,  ay],
                     [az,  0.0, -ax],
                     [-ay, ax,  0.0]], dtype=float)

def R_b_to_n(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [ cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [ spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [   -sth,              cth*sphi,                 cth*cphi]
    ], dtype=float)

def T_omega(phi: float, theta: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    if abs(cth) < 1e-6:
        cth = np.sign(cth) * 1e-6
    return np.array([
        [1.0, sphi*sth/cth,  cphi*sth/cth],
        [0.0,        cphi,         -sphi],
        [0.0, sphi/cth,      cphi/cth]
    ], dtype=float)

def J_eta(eta: np.ndarray) -> np.ndarray:
    phi, theta, psi = float(eta[3]), float(eta[4]), float(eta[5])
    R = R_b_to_n(phi, theta, psi)
    T = T_omega(phi, theta)
    J = np.zeros((6, 6), dtype=float)
    J[0:3, 0:3] = R
    J[3:6, 3:6] = T
    return J

def M_RB(params: dict) -> np.ndarray:
    m = float(params["m"])
    Ixx, Iyy, Izz = float(params["I_xx"]), float(params["I_yy"]), float(params["I_zz"])
    rg = np.array([float(params["x_cg"]), float(params["y_cg"]), float(params["z_cg"])], dtype=float)

    I = np.diag([Ixx, Iyy, Izz]).astype(float)
    Srg = skew(rg)

    M = np.zeros((6, 6), dtype=float)
    M[0:3, 0:3] = m * np.eye(3)
    M[0:3, 3:6] = -m * Srg
    M[3:6, 0:3] =  m * Srg
    M[3:6, 3:6] = I
    return M

def M_A(params: dict) -> np.ndarray:
    Xdu = float(params["X_dot_u"])
    Ydv = float(params["Y_dot_v"])
    Zdw = float(params["Z_dot_w"])
    Kdp = float(params["K_dot_p"])
    Mdq = float(params["M_dot_q"])
    Ndr = float(params["N_dot_r"])
    return -np.diag([Xdu, Ydv, Zdw, Kdp, Mdq, Ndr]).astype(float)

def M_total(params: dict) -> np.ndarray:
    return M_RB(params) + M_A(params)

def C_RB(nu: np.ndarray, params: dict) -> np.ndarray:
    u, v, w, p, q, r = [float(nu[i]) for i in range(6)]
    m = float(params["m"])
    Ixx = float(params["I_xx"])
    Iyy = float(params["I_yy"])
    Izz = float(params["I_zz"])
    xg = float(params["x_cg"])
    yg = float(params["y_cg"])
    zg = float(params["z_cg"])

    C = np.zeros((6, 6), dtype=float)

    C[0, 1] = -m * r
    C[0, 2] =  m * q
    C[1, 0] =  m * r
    C[1, 2] = -m * p
    C[2, 0] = -m * q
    C[2, 1] =  m * p

    C[0, 3] =  m * (q * yg + r * zg)
    C[0, 4] = -m * (q * xg)
    C[0, 5] = -m * (r * xg)

    C[1, 3] = -m * (p * yg)
    C[1, 4] =  m * (p * xg + r * zg)
    C[1, 5] = -m * (r * yg)

    C[2, 3] = -m * (p * zg)
    C[2, 4] = -m * (q * zg)
    C[2, 5] =  m * (p * xg + q * yg)

    C[3, 0] = -m * (q * yg + r * zg)
    C[3, 1] =  m * (p * yg)
    C[3, 2] =  m * (p * zg)

    C[4, 0] =  m * (q * xg)
    C[4, 1] = -m * (p * xg + r * zg)
    C[4, 2] =  m * (q * zg)

    C[5, 0] =  m * (r * xg)
    C[5, 1] =  m * (r * yg)
    C[5, 2] = -m * (p * xg + q * yg)

    C[3, 4] =  Izz * r
    C[3, 5] = -Iyy * q
    C[4, 3] = -Izz * r
    C[4, 5] =  Ixx * p
    C[5, 3] =  Iyy * q
    C[5, 4] = -Ixx * p

    return C

def C_A(nu: np.ndarray, params: dict) -> np.ndarray:
    u, v, w, p, q, r = [float(nu[i]) for i in range(6)]
    Xdu = float(params["X_dot_u"])
    Ydv = float(params["Y_dot_v"])
    Zdw = float(params["Z_dot_w"])
    Kdp = float(params["K_dot_p"])
    Mdq = float(params["M_dot_q"])
    Ndr = float(params["N_dot_r"])

    a1 = Xdu * u
    a2 = Ydv * v
    a3 = Zdw * w
    b1 = Kdp * p
    b2 = Mdq * q
    b3 = Ndr * r

    C = np.zeros((6, 6), dtype=float)
    C[0, 4] = -a3
    C[0, 5] =  a2
    C[1, 3] =  a3
    C[1, 5] = -a1
    C[2, 3] = -a2
    C[2, 4] =  a1
    C[3, 1] = -a3
    C[3, 2] =  a2
    C[3, 4] = -b3
    C[3, 5] =  b2
    C[4, 0] =  a3
    C[4, 2] = -a1
    C[4, 3] =  b3
    C[4, 5] = -b1
    C[5, 0] = -a2
    C[5, 1] =  a1
    C[5, 3] = -b2
    C[5, 4] =  b1
    return C

def C_total(nu: np.ndarray, params: dict, include_ca: bool) -> np.ndarray:
    C = C_RB(nu, params)
    if include_ca:
        C = C + C_A(nu, params)
    return C

def D_nu(nu: np.ndarray, params: dict) -> np.ndarray:
    u, v, w, p, q, r = [float(x) for x in nu]
    lin = np.array([
        float(params["X_u"]), float(params["Y_v"]), float(params["Z_w"]),
        float(params["K_p"]), float(params["M_q"]), float(params["N_r"]),
    ], dtype=float)
    quad = np.array([
        float(params["X_uu"]), float(params["Y_vv"]), float(params["Z_ww"]),
        float(params["K_pp"]), float(params["M_qq"]), float(params["N_rr"]),
    ], dtype=float)
    abs_nu = np.array([abs(u), abs(v), abs(w), abs(p), abs(q), abs(r)], dtype=float)
    diag_entries = lin + quad * abs_nu
    return -np.diag(diag_entries)

def g_eta(eta: np.ndarray, params: dict) -> np.ndarray:
    phi   = float(eta[3])
    theta = float(eta[4])

    g0 = 9.8
    m = float(params["m"])
    W = m * g0
    B = float(params["B"])

    x_G = float(params["x_cg"]); y_G = float(params["y_cg"]); z_G = float(params["z_cg"])
    x_B = float(params["x_cb"]); y_B = float(params["y_cb"]); z_B = float(params["z_cb"])

    WB    = W - B
    xW_xB = x_G * W - x_B * B
    yW_yB = y_G * W - y_B * B
    zW_zB = z_G * W - z_B * B

    cth, sth   = np.cos(theta), np.sin(theta)
    cphi, sphi = np.cos(phi), np.sin(phi)

    return np.array([
        WB * sth,
        -WB * cth * sphi,
        -WB * cth * cphi,
        -yW_yB * cth * cphi + zW_zB * cth * sphi,
         zW_zB * sth + xW_xB * cth * cphi,
        -xW_xB * cth * sphi - yW_yB * sth
    ], dtype=float)

def forward_integrate_6dof_full(
    t: np.ndarray,
    nu0: np.ndarray,
    eta0: np.ndarray,
    tau: np.ndarray,
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
    nu_dot0: np.ndarray | None = None,
):
    """
    Single-seed Euler integration over full time vector.

    If nu_dot0 is provided, it's used ONLY for the first Euler step (k=0->1).
    After that, nu_dot is computed from the model each step.
    """
    N = len(t)
    nu_pred = np.zeros((N, 6), dtype=float)
    eta_pred = np.zeros((N, 6), dtype=float)
    nu_dot_pred = np.zeros((N, 6), dtype=float)

    nu_pred[0, :] = np.asarray(nu0, dtype=float).reshape(6,)
    eta_pred[0, :] = np.asarray(eta0, dtype=float).reshape(6,)

    M = M_total(params)

    # set initial nu_dot (optional)
    if nu_dot0 is not None:
        nu_dot_pred[0, :] = np.asarray(nu_dot0, dtype=float).reshape(6,)
    else:
        nu_k = nu_pred[0, :]
        eta_k = eta_pred[0, :]
        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)
        rhs = tau[0, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot_pred[0, :] = np.linalg.solve(M, rhs)

    for k in range(1, N):
        dt = DT_FIXED if FORCE_FIXED_DT else float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt <= 0.0:
            nu_pred[k, :] = nu_pred[k - 1, :]
            eta_pred[k, :] = eta_pred[k - 1, :]
            nu_dot_pred[k, :] = 0.0
            continue

        nu_k  = nu_pred[k - 1, :]
        eta_k = eta_pred[k - 1, :]

        # compute nu_dot from dynamics
        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)
        rhs = tau[k - 1, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot = np.linalg.solve(M, rhs)

        # Euler: nu_{k+1}
        nu_kp1 = nu_k + nu_dot * dt
        nu_pred[k, :] = nu_kp1
        nu_dot_pred[k, :] = nu_dot

        # kinematics: eta_{k+1}
        J = J_eta(eta_k)
        eta_dot = J @ nu_kp1
        eta_pred[k, :] = eta_k + eta_dot * dt

    return nu_pred, eta_pred, nu_dot_pred



def residual_stats(residual: np.ndarray):
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return np.nan, np.nan
    mu = float(np.mean(r))
    rmse = float(np.sqrt(np.mean(r**2)))
    return mu, rmse

def error_metrics(y_meas: np.ndarray, y_pred: np.ndarray):
    """
    Returns RMSE (units), NRMSE_sigma (dimensionless), and %RMSE (percent),
    computed on samples where both y_meas and y_pred are finite.
    """
    m = np.isfinite(y_meas) & np.isfinite(y_pred)
    if np.sum(m) < 5:
        return dict(n=int(np.sum(m)), rmse=np.nan, nrmse_sigma=np.nan, prmse=np.nan)

    e = y_meas[m] - y_pred[m]
    rmse = float(np.sqrt(np.mean(e**2)))

    sig = float(np.std(y_meas[m], ddof=0))
    nrmse_sigma = rmse / sig if sig > 1e-12 else np.nan

    rms_y = float(np.sqrt(np.mean(y_meas[m]**2)))
    prmse = 100.0 * (rmse / rms_y) if rms_y > 1e-12 else np.nan

    return dict(n=int(np.sum(m)), rmse=rmse, nrmse_sigma=nrmse_sigma, prmse=prmse)

def error_metrics_masked(y_meas: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(y_meas) & np.isfinite(y_pred)
    if np.sum(m) < 5:
        return dict(n=int(np.sum(m)), rmse=np.nan, nrmse_sigma=np.nan, prmse=np.nan)

    e = y_meas[m] - y_pred[m]
    rmse = float(np.sqrt(np.mean(e**2)))

    sig = float(np.std(y_meas[m], ddof=0))
    nrmse_sigma = rmse / sig if sig > 1e-12 else np.nan

    rms_y = float(np.sqrt(np.mean(y_meas[m]**2)))
    prmse = 100.0 * (rmse / rms_y) if rms_y > 1e-12 else np.nan

    return dict(n=int(np.sum(m)), rmse=rmse, nrmse_sigma=nrmse_sigma, prmse=prmse)

# ============================================================
# === 4) Run predictions (store eta) =========================
# ============================================================
enabled_labels = []
if PLOT_MLE: enabled_labels.append("MLE")
if PLOT_SIM_MLE: enabled_labels.append("SIM_MLE")
if PLOT_SIM_TRUTH: enabled_labels.append("SIM_TRUTH")
if PLOT_TEST_PARAMS_1: enabled_labels.append("TEST_1")
if PLOT_TEST_PARAMS_2: enabled_labels.append("TEST_2")
if PLOT_MAP: enabled_labels.append("MAP")
if PLOT_HMC_TEST_PARAMS_1: enabled_labels.append("HMC_TEST_1")

if len(enabled_labels) == 0:
    raise ValueError("No prediction methods enabled.")

pred_eta = {}

residual_eta = {lab: {} for lab in enabled_labels}
stats_eta    = {lab: {} for lab in enabled_labels}
metrics_eta  = {lab: {} for lab in enabled_labels}

for label in enabled_labels:
    p = PARAM_SETS[label]

    nu_pred_all, eta_pred_all, nu_dot_pred_all = forward_integrate_6dof_full(
        t=t,
        nu0=nu_meas[0, :],
        eta0=eta_meas[0, :],
        nu_dot0= None,   # optional (your request)
        tau=tau,
        params=p,
        include_ca=INCLUDE_CA,
        include_g=INCLUDE_G,
    )

    pred_eta[label] = eta_pred_all


    for cname, idx in EVAL_COMPONENTS.items():
        y_meas = eta_meas[:, idx].astype(float)
        y_pred = eta_pred_all[:, idx].astype(float)

        base_mask = time_window_mask

        if cname in ANGLE_COMPONENTS:
            ym = wrap_to_pi(y_meas.copy())
            yp = wrap_to_pi(y_pred.copy())
            r  = wrap_to_pi(ym - yp)
            residual_eta[label][cname] = np.where(np.isfinite(yp) & base_mask, r, np.nan)
            metrics_eta[label][cname] = error_metrics(ym[base_mask], yp[base_mask])
        else:
            r = y_meas - y_pred
            residual_eta[label][cname] = np.where(np.isfinite(y_pred) & base_mask, r, np.nan)
            metrics_eta[label][cname] = error_metrics(y_meas[base_mask], y_pred[base_mask])

        stats_eta[label][cname] = residual_stats(residual_eta[label][cname])

# ============================================================
# === 5) Plots: time-series (x, y, z, psi) ===================
# ============================================================
TITLE_FS  = 18
AXIS_FS   = 15
TICK_FS   = 12
LEGEND_FS = 11

MEAS_LW   = 3.0
PRED_LW   = 2.0

METHOD_COLORS = {
    "MLE": "#ff7f0e",
    "MAP": "#2ca02c",
    "SIM_TRUTH": "#d62728",
    "SIM_MLE": "#9467bd",
    "TEST_1": "#8c564b",
    "TEST_2": "#e377c2",
    "HMC_TEST_1": "#7f7f7f",
}

def color_for(label: str):
    return METHOD_COLORS.get(label, None)

plot_mask = time_window_mask.copy()
if pose_valid is not None:
    plot_mask &= (pose_valid > 0.5)

t_plot = np.where(plot_mask, t_rel, np.nan)

fig = plt.figure(figsize=(14, 10))
fig.suptitle("6DOF Position/Attitude Post-Check", fontsize=TITLE_FS, fontweight="bold", y=0.98)

panels = [("x", "X [m]"), ("y", "Y [m]"), ("z", "Z [m]"), ("psi", "Yaw ψ [rad]")]
for k, (cname, ylabel) in enumerate(panels, start=1):
    idx = EVAL_COMPONENTS[cname]
    ax = plt.subplot(4, 1, k)

    y_meas = eta_meas[:, idx].astype(float)
    if cname in ANGLE_COMPONENTS:
        y_meas = wrap_to_pi(y_meas)

    ax.plot(t_plot, y_meas, linewidth=MEAS_LW, color="#1f77b4", label=f"{cname} measured", zorder=4)

    for label in enabled_labels:
        y_pred = pred_eta[label][:, idx].astype(float)
        if cname in ANGLE_COMPONENTS:
            y_pred = wrap_to_pi(y_pred)
        ax.plot(t_plot, y_pred, linewidth=PRED_LW, color=color_for(label), alpha=0.95,
                label=f"{cname} predicted ({label})", zorder=6)
    #
    # if ENABLE_HMC_BAND and (eta_p05 is not None):
    #     lo = eta_p05[:, idx].astype(float)
    #     hi = eta_p95[:, idx].astype(float)
    #     med = eta_p50[:, idx].astype(float)
    #     if cname in ANGLE_COMPONENTS:
    #         lo = wrap_to_pi(lo); hi = wrap_to_pi(hi); med = wrap_to_pi(med)
    #     ax.fill_between(t_plot, lo, hi, alpha=0.25, label="HMC 90% band")
    #     ax.plot(t_plot, med, linestyle="--", linewidth=2.2, label="HMC median")
    #
    # ax.set_ylabel(ylabel, fontsize=AXIS_FS, fontweight="bold")
    # ax.grid(True, alpha=0.35)
    # ax.tick_params(axis="both", labelsize=TICK_FS)
    # if k == 1:
    #     ax.legend(loc="best", fontsize=LEGEND_FS, framealpha=0.95, ncol=2)

plt.xlabel("Time [s]", fontsize=AXIS_FS, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.show()

# ============================================================
# === 5b) Plot: XY track overlay =============================
# ============================================================
x_meas = eta_meas[:, 0].astype(float)
y_meas = eta_meas[:, 1].astype(float)

x_meas_p = np.full_like(x_meas, np.nan)
y_meas_p = np.full_like(y_meas, np.nan)
x_meas_p[plot_mask] = x_meas[plot_mask]
y_meas_p[plot_mask] = y_meas[plot_mask]

valid_idx = np.where(np.isfinite(x_meas_p) & np.isfinite(y_meas_p))[0]
x0 = y0 = 0.0
if valid_idx.size > 0:
    i0 = int(valid_idx[0])
    x0, y0 = float(x_meas_p[i0]), float(y_meas_p[i0])
    x_meas_p = x_meas_p - x0
    y_meas_p = y_meas_p - y0

plt.figure(figsize=(8.5, 8.5))
plt.plot(x_meas_p, y_meas_p, linewidth=3.0, color="#1f77b4", label="Truth (measured)")

for label in enabled_labels:
    x_pred = pred_eta[label][:, 0].astype(float)
    y_pred = pred_eta[label][:, 1].astype(float)

    x_pred_p = np.full_like(x_pred, np.nan)
    y_pred_p = np.full_like(y_pred, np.nan)
    x_pred_p[plot_mask] = x_pred[plot_mask]
    y_pred_p[plot_mask] = y_pred[plot_mask]

    x_pred_p = x_pred_p - x0
    y_pred_p = y_pred_p - y0

    plt.plot(x_pred_p, y_pred_p, linewidth=2.0, color=color_for(label), alpha=0.95, label=f"Pred ({label})")

plt.xlabel("X [m]", fontsize=14)
plt.ylabel("Y [m]", fontsize=14)
plt.title("XY Track: Truth vs Predicted", fontsize=16)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend(framealpha=0.95)
plt.tight_layout()
plt.show()

# ============================================================
# === 5b-ANIM) Animate: XY track overlay =====================
# ============================================================
# === SAFE LOW-RES DEFAULTS (tune later) ===
ANIM_METHOD = "MLE"

ANIM_STRIDE = 20         # big downsample (10–30 is typical)
ANIM_INTERVAL_MS = 60     # slower playback while previewing
ANIM_TAIL = None           # bounded draw cost (None is expensive)
MAX_FRAMES = 600          # hard cap so it can’t explode

SAVE_ANIM = True
SHOW_ANIM = False         # IMPORTANT: don't show if you're saving
SAVE_PATH = "xy_dead_reckoning_full.mp4"   # start with gif if no ffmpeg
SAVE_DPI = 80             # low-res raster
SAVE_FPS = 30             # low fps is fine for slides

if ANIM_METHOD not in enabled_labels:
    raise ValueError(f"ANIM_METHOD='{ANIM_METHOD}' not enabled. Enabled: {enabled_labels}")

# Build plotted (masked) signals
x_meas = eta_meas[:, 0].astype(float)
y_meas = eta_meas[:, 1].astype(float)

x_pred = pred_eta[ANIM_METHOD][:, 0].astype(float)
y_pred = pred_eta[ANIM_METHOD][:, 1].astype(float)

# Apply same plot mask you already use (pose_valid + time window)
mask_xy = plot_mask & np.isfinite(x_meas) & np.isfinite(y_meas) & np.isfinite(x_pred) & np.isfinite(y_pred)

# Extract only valid indices (keeps gaps out of the animation)
idx = np.where(mask_xy)[0]

# If nothing valid, bail
if idx.size < 5:
    raise RuntimeError("Not enough valid XY samples to animate (check plot_mask / validity gating).")

# Re-zero to initial point for clean overlay (same as your static plot logic)
i0 = int(idx[0])
x0, y0 = float(x_meas[i0]), float(y_meas[i0])

xm = x_meas[idx] - x0
ym = y_meas[idx] - y0
xp = x_pred[idx] - x0
yp = y_pred[idx] - y0

# Downsample for speed
xm = xm[::ANIM_STRIDE]
ym = ym[::ANIM_STRIDE]
xp = xp[::ANIM_STRIDE]
yp = yp[::ANIM_STRIDE]

# Set plot limits with padding
allx = np.r_[xm, xp]
ally = np.r_[ym, yp]
pad = 0.5
xmin, xmax = np.nanmin(allx) - pad, np.nanmax(allx) + pad
ymin, ymax = np.nanmin(ally) - pad, np.nanmax(ally) + pad

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.set_title(f"XY Dead Reckoning Animation: Truth vs {ANIM_METHOD}", fontsize=16)
ax.set_xlabel("X [m]", fontsize=14)
ax.set_ylabel("Y [m]", fontsize=14)
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Two animated lines + current-point markers
truth_line, = ax.plot([], [], linewidth=3.0, label="Truth (measured)")
pred_line,  = ax.plot([], [], linewidth=2.0, alpha=0.95, label=f"Pred ({ANIM_METHOD})")

truth_pt, = ax.plot([], [], marker="o", markersize=6)
pred_pt,  = ax.plot([], [], marker="o", markersize=6)

ax.legend(framealpha=0.95)

def init():
    truth_line.set_data([], [])
    pred_line.set_data([], [])
    truth_pt.set_data([], [])
    pred_pt.set_data([], [])
    return truth_line, pred_line, truth_pt, pred_pt

def update(frame):
    # frame is last index to draw (inclusive-ish)
    k = frame

    if ANIM_TAIL is None:
        i_start = 0
    else:
        i_start = max(0, k - int(ANIM_TAIL))

    # segment to draw
    xs_t = xm[i_start:k]
    ys_t = ym[i_start:k]
    xs_p = xp[i_start:k]
    ys_p = yp[i_start:k]

    truth_line.set_data(xs_t, ys_t)
    pred_line.set_data(xs_p, ys_p)

    # current markers (use last available point)
    if k > 0:
        truth_pt.set_data([xm[k-1]], [ym[k-1]])
        pred_pt.set_data([xp[k-1]], [yp[k-1]])

    return truth_line, pred_line, truth_pt, pred_pt

n_frames = min(len(xm), MAX_FRAMES)
anim = FuncAnimation(
    fig,
    update,
    frames=range(1, n_frames + 1),
    init_func=init,
    interval=ANIM_INTERVAL_MS,
    blit=True,
    repeat=False,
)

if SHOW_ANIM:
    plt.show()

if SAVE_ANIM:
    if SAVE_PATH.lower().endswith(".gif"):
        anim.save(SAVE_PATH, writer="pillow", fps=SAVE_FPS, dpi=SAVE_DPI)

    elif SAVE_PATH.lower().endswith(".mp4"):
        writer = FFMpegWriter(
            fps=SAVE_FPS,
            codec="libx264",
            bitrate=1200,                     # keep it small
            extra_args=["-pix_fmt", "yuv420p"] # PowerPoint-friendly
        )
        anim.save(SAVE_PATH, writer=writer, dpi=SAVE_DPI)

    else:
        raise ValueError("SAVE_PATH must end with .mp4 or .gif")

    print(f"[INFO] Saved animation to: {SAVE_PATH}")
    plt.close(fig)

# ============================================================
# === 5c) Print error metrics per component ==================
# ============================================================
print("\n=== Position/Attitude Error Metrics (time-windowed) ===")
for label in enabled_labels:
    print(f"\n[{label}]")
    for cname, idx in EVAL_COMPONENTS.items():
        mu, _ = stats_eta[label][cname]
        m = metrics_eta[label][cname]
        unit = "rad" if cname in ANGLE_COMPONENTS else "m"
        print(
            f"  {cname:>4s}: "
            f"mean={mu:+.4f} {unit}, "
            f"rmse={m['rmse']:.4f} {unit}, "
            f"nrmseσ={m['nrmse_sigma']:.3f}, "
            f"%rmse={m['prmse']:.1f}%  (N={m['n']})"
        )
print("=======================================================\n")

# ============================================================
# === Amplitude-dependent error metrics (by ||tau|| bins) =====
# ============================================================
tau_norm = np.linalg.norm(tau, axis=1).astype(float)

bins = [
    ("||τ|| ≤ 25", tau_norm <= 25),
    ("25 < ||τ|| ≤ 50", (tau_norm > 25) & (tau_norm <= 50)),
    ("||τ|| > 50", tau_norm > 50),
]

print("\n=== Amplitude-dependent error metrics (by ||tau||) ===")
for label in enabled_labels:
    print(f"\n[{label}]")
    for cname, idx in EVAL_COMPONENTS.items():
        unit = "rad" if cname in ANGLE_COMPONENTS else "m"
        y_meas = eta_meas[:, idx].astype(float)
        y_pred = pred_eta[label][:, idx].astype(float)

        if cname in ANGLE_COMPONENTS:
            y_meas = wrap_to_pi(y_meas)
            y_pred = wrap_to_pi(y_pred)

        print(f"  component: {cname} [{unit}]")
        for name, msk in bins:
            mm = error_metrics_masked(y_meas, y_pred, msk & time_window_mask)
            print(
                f"    {name:14s}  N={mm['n']:5d}  "
                f"RMSE={mm['rmse']:.4f} {unit}  "
                f"NRMSEσ={mm['nrmse_sigma']:.3f}  "
                f"%RMSE={mm['prmse']:.1f}%"
            )
print("====================================================\n")

# ============================================================
# === 6) dt sanity print =====================================
# ============================================================
dt_vec = np.diff(t)
print("dt stats:", np.nanmin(dt_vec), np.nanmean(dt_vec), np.nanmax(dt_vec))
