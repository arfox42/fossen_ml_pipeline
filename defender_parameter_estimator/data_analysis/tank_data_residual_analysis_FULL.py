#!/usr/bin/env python3
"""
post_check_full6dof_single_dof.py
-------------------------------------
Forward-integrates full 6DOF, but plots ONLY the selected ACTIVE_DOF channel
(velocity + residual), plus:
    - optional segmented integration (NaN/dropout aware)
    - optional no-integration accel consistency check
    - optional inverse dynamics tau residual diagnostics
    - optional HMC posterior predictive band (segmented + filtered)

Core model:
    M nu_dot + C(nu)nu + D(nu)nu + g(eta) = tau
    eta_dot = J(eta) nu
"""

import numpy as np
import matplotlib.pyplot as plt
import torch  # only needed if you later enable HMC band loading

# ============================================================
# === 0) Config / Flags ======================================
# ============================================================

# ------------------------------------------------------------
# 0a) Paths (you set these manually)
# ------------------------------------------------------------
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Coupled Maneuvers/defender_data_teleop_circle.csv"
HMC_SAMPLES_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_yaw_samples_2.pt"

# --- NaN / dropout handling ---
SEGMENTED_INTEGRATION = False          # integrate only over valid segments; NaNs stay as gaps
USE_VALIDITY_FLAGS = True             # gate using pose/twist/wrench valid fields if present
REQUIRE_POSE_VALID = False
REQUIRE_TWIST_VALID = True
REQUIRE_WRENCH_VALID = True

MAX_ALLOWED_AGE_S = None              # e.g. 0.05, or None
MIN_SEG_LEN = 50                      # samples

# ------------------------------------------------------------
# 0b) Dataset / DOF
# ------------------------------------------------------------
ACTIVE_DOF = "N"   # "X","Y","Z","K","M","N"

# ------------------------------------------------------------
# 0c) Physics toggles
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
FORCE_FIXED_DT = False
DT_FIXED = 0.01

# ------------------------------------------------------------
# 0e) Velocity forward-integration overlays (main 3-panel plot)
# ------------------------------------------------------------
PLOT_SIM_MLE   = False
PLOT_SIM_TRUTH = False
PLOT_MLE       = True

PLOT_TEST_PARAMS_1 = False
PLOT_TEST_PARAMS_2 = False

PLOT_MAP = True
PLOT_HMC_TEST_PARAMS_1 = False

# ------------------------------------------------------------
# 0f) HMC posterior predictive band (6DOF forward integration)
# ------------------------------------------------------------
ENABLE_HMC_BAND = True
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
HMC_BAND_PARAM_SUBSET = HMC_BAND_PARAMS_BY_DOF[ACTIVE_DOF] + HMC_BAND_GLOBAL_EXTRAS

# ------------------------------------------------------------
# 0g) Optional accel consistency check (no integration)
# CSV ν,η,τ -> predicted ν̇ (compare to ν̇ in CSV)
# ------------------------------------------------------------
PLOT_ACCEL_CHECK       = False
PLOT_ACCEL_SIM         = False
PLOT_ACCEL_MLE         = False
PLOT_ACCEL_TEST_1      = False
PLOT_ACCEL_TEST_2      = False
PLOT_ACCEL_MAP         = False
PLOT_ACCEL_HMC_TEST_1  = False

# ------------------------------------------------------------
# 0h) Optional tau residual diagnostics (inverse dynamics)
# Uses measured (nu, nu_dot, eta) and computes:
# tau_hat = M*nu_dot + C*nu + D*nu + g
# then compares commanded tau (CSV) vs tau_hat
# ------------------------------------------------------------
PLOT_TAU_RESIDUAL_DIAGNOSTICS = False
TAU_RESIDUAL_LABELS = []
if PLOT_SIM_TRUTH:
    TAU_RESIDUAL_LABELS.append("SIM_TRUTH")
TAU_RESIDUAL_PCT = 95

# ------------------------------------------------------------
# 0i) quick config printouts
# ------------------------------------------------------------
print(f"[config] ACTIVE_DOF={ACTIVE_DOF}")
print(f"[config] INCLUDE_CA={INCLUDE_CA}, INCLUDE_G={INCLUDE_G}")
print(f"[config] SEGMENTED_INTEGRATION={SEGMENTED_INTEGRATION}, USE_VALIDITY_FLAGS={USE_VALIDITY_FLAGS}")
print(f"[config] HMC_BASELINE={HMC_BASELINE}, ENABLE_HMC_BAND={ENABLE_HMC_BAND}")
print(f"[config] HMC_BAND_PARAM_SUBSET={HMC_BAND_PARAM_SUBSET}")
print(f"[config] TAU_RESIDUAL_LABELS={TAU_RESIDUAL_LABELS}, TAU_RESIDUAL_PCT={TAU_RESIDUAL_PCT}")

# ============================================================
# === Index maps =============================================
# ============================================================
DOF_INDEX = {
    "X": 0,  # u
    "Y": 1,  # v
    "Z": 2,  # w
    "K": 3,  # p
    "M": 4,  # q
    "N": 5,  # r
}

DOF_META = {
    "X": {"name": "Surge", "sym": "u", "tau": "X", "unit": "m/s",   "tau_unit": "N",   "ylim": (-2, 2)},
    "Y": {"name": "Sway",  "sym": "v", "tau": "Y", "unit": "m/s",   "tau_unit": "N",   "ylim": (-2, 2)},
    "Z": {"name": "Heave", "sym": "w", "tau": "Z", "unit": "m/s",   "tau_unit": "N",   "ylim": (-2, 2)},
    "K": {"name": "Roll",  "sym": "p", "tau": "K", "unit": "rad/s", "tau_unit": "N·m", "ylim": (-3, 3)},
    "M": {"name": "Pitch", "sym": "q", "tau": "M", "unit": "rad/s", "tau_unit": "N·m", "ylim": (-3, 3)},
    "N": {"name": "Yaw",   "sym": "r", "tau": "N", "unit": "rad/s", "tau_unit": "N·m", "ylim": (-3, 3)},
}

ACCEL_META = {
    "X": {"name": "Surge", "sym": "u̇", "unit": "m/s²"},
    "Y": {"name": "Sway",  "sym": "v̇", "unit": "m/s²"},
    "Z": {"name": "Heave", "sym": "ẇ", "unit": "m/s²"},
    "K": {"name": "Roll",  "sym": "ṗ", "unit": "rad/s²"},
    "M": {"name": "Pitch", "sym": "q̇", "unit": "rad/s²"},
    "N": {"name": "Yaw",   "sym": "ṙ", "unit": "rad/s²"},
}

# ============================================================
# === 1) Load CSV ============================================
# ============================================================
data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)
if data.ndim != 2 or data.shape[1] < 25:
    raise RuntimeError(f"CSV seems malformed or too few columns: shape={data.shape}")

t = data[:, 0].astype(float)
t_rel = t - t[0]

# ν_dot (1..6)
nu_dot_meas = np.vstack([data[:, i].astype(float) for i in range(1, 7)]).T

# ν (7..12)
nu_meas = np.vstack([data[:, i].astype(float) for i in range(7, 13)]).T

# η (13..18)
eta_meas = np.vstack([data[:, i].astype(float) for i in range(13, 19)]).T

# τ (19..24)
tau = np.vstack([data[:, i].astype(float) for i in range(19, 25)]).T

if USE_TAU_SCALE:
    print(f"[INFO] Scaling tau by factor {TAU_SCALE}")
    tau = TAU_SCALE * tau

# Optional validity columns (25..30) if present
pose_valid = pose_age = twist_valid = twist_age = wrench_valid = wrench_age = None
if data.shape[1] >= 31:
    pose_valid   = data[:, 25].astype(float)
    pose_age     = data[:, 26].astype(float)
    twist_valid  = data[:, 27].astype(float)
    twist_age    = data[:, 28].astype(float)
    wrench_valid = data[:, 29].astype(float)
    wrench_age   = data[:, 30].astype(float)

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

def forward_integrate_6dof_segment(
    t_seg: np.ndarray,
    nu0: np.ndarray,
    eta0: np.ndarray,
    tau_seg: np.ndarray,
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
):
    N = len(t_seg)
    nu_pred = np.zeros((N, 6), dtype=float)
    eta_pred = np.zeros((N, 6), dtype=float)
    nu_dot_pred = np.zeros((N, 6), dtype=float)

    nu_pred[0, :] = np.asarray(nu0, dtype=float).reshape(6,)
    eta_pred[0, :] = np.asarray(eta0, dtype=float).reshape(6,)

    M = M_total(params)

    for k in range(1, N):
        dt = DT_FIXED if FORCE_FIXED_DT else float(t_seg[k] - t_seg[k - 1])

        if not np.isfinite(dt) or dt <= 0.0:
            nu_pred[k, :] = nu_pred[k - 1, :]
            eta_pred[k, :] = eta_pred[k - 1, :]
            nu_dot_pred[k, :] = 0.0
            continue

        nu_k = nu_pred[k - 1, :]
        eta_k = eta_pred[k - 1, :]

        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)

        rhs = tau_seg[k - 1, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot = np.linalg.solve(M, rhs)

        nu_kp1 = nu_k + nu_dot * dt
        nu_pred[k, :] = nu_kp1
        nu_dot_pred[k, :] = nu_dot

        J = J_eta(eta_k)
        eta_dot = J @ nu_kp1
        eta_pred[k, :] = eta_k + eta_dot * dt

    return nu_pred, eta_pred, nu_dot_pred

def segmented_forward_prediction(
    t: np.ndarray,
    nu_meas: np.ndarray,
    eta_meas: np.ndarray,
    tau: np.ndarray,
    params: dict,
    segs: list,
    include_ca: bool,
    include_g: bool,
):
    N = len(t)
    nu_pred_all = np.full((N, 6), np.nan, dtype=float)

    for (a, b) in segs:
        nu_pred_seg, _, _ = forward_integrate_6dof_segment(
            t_seg=t[a:b],
            nu0=nu_meas[a, :],
            eta0=eta_meas[a, :],
            tau_seg=tau[a:b, :],
            params=params,
            include_ca=include_ca,
            include_g=include_g,
        )
        nu_pred_all[a:b, :] = nu_pred_seg

    return nu_pred_all

def residual_stats(residual: np.ndarray):
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return np.nan, np.nan
    mu = float(np.mean(r))
    rmse = float(np.sqrt(np.mean(r**2)))
    return mu, rmse

def fmt_mu_rmse(mu, rmse, unit="", nd=4):
    if (mu is None) or (rmse is None) or (not np.isfinite(mu)) or (not np.isfinite(rmse)):
        return "mean=nan, rmse=nan"
    if unit:
        return f"mean={mu:+.{nd}f} {unit}, rmse={rmse:.{nd}f} {unit}"
    return f"mean={mu:+.{nd}f}, rmse={rmse:.{nd}f}"


def error_metrics(y_meas: np.ndarray, y_pred: np.ndarray):
    """
    Returns RMSE (units), NRMSE_sigma (dimensionless), and %RMSE (percent),
    computed on samples where both y_meas and y_pred are finite.
    """
    m = np.isfinite(y_meas) & np.isfinite(y_pred)
    if np.sum(m) < 5:
        return dict(rmse=np.nan, nrmse_sigma=np.nan, prmse=np.nan)

    e = y_meas[m] - y_pred[m]

    rmse = float(np.sqrt(np.mean(e**2)))

    sig = float(np.std(y_meas[m], ddof=0))
    nrmse_sigma = rmse / sig if sig > 1e-12 else np.nan

    rms_y = float(np.sqrt(np.mean(y_meas[m]**2)))
    prmse = 100.0 * (rmse / rms_y) if rms_y > 1e-12 else np.nan

    return dict(rmse=rmse, nrmse_sigma=nrmse_sigma, prmse=prmse)

def error_metrics_masked(y_meas: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """
    Same metrics, but only on samples where mask is True AND both signals are finite.
    """
    m = mask & np.isfinite(y_meas) & np.isfinite(y_pred)
    if np.sum(m) < 5:
        return dict(n=np.sum(m), rmse=np.nan, nrmse_sigma=np.nan, prmse=np.nan)

    e = y_meas[m] - y_pred[m]
    rmse = float(np.sqrt(np.mean(e**2)))

    sig = float(np.std(y_meas[m], ddof=0))
    nrmse_sigma = rmse / sig if sig > 1e-12 else np.nan

    rms_y = float(np.sqrt(np.mean(y_meas[m]**2)))
    prmse = 100.0 * (rmse / rms_y) if rms_y > 1e-12 else np.nan

    return dict(n=int(np.sum(m)), rmse=rmse, nrmse_sigma=nrmse_sigma, prmse=prmse)


def predict_nu_dot_from_csv_states_masked(
    nu_meas: np.ndarray,
    eta_meas: np.ndarray,
    tau: np.ndarray,
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
    mask: np.ndarray = None,
) -> np.ndarray:
    N = nu_meas.shape[0]
    nu_dot_hat = np.full((N, 6), np.nan, dtype=float)
    M = M_total(params)

    if mask is None:
        mask = np.ones(N, dtype=bool)

    idxs = np.where(mask)[0]
    for k in idxs:
        nu_k = nu_meas[k, :]
        eta_k = eta_meas[k, :]
        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)
        rhs = tau[k, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot_hat[k, :] = np.linalg.solve(M, rhs)

    return nu_dot_hat

def predict_tau_from_csv_states_masked(
    nu_meas: np.ndarray,
    nu_dot_meas: np.ndarray,
    eta_meas: np.ndarray,
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
    mask: np.ndarray = None,
) -> np.ndarray:
    N = nu_meas.shape[0]
    tau_hat = np.full((N, 6), np.nan, dtype=float)
    M = M_total(params)

    if mask is None:
        mask = np.ones(N, dtype=bool)

    idxs = np.where(mask)[0]
    for k in idxs:
        nu_k = nu_meas[k, :]
        nud_k = nu_dot_meas[k, :]
        eta_k = eta_meas[k, :]
        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)
        tau_hat[k, :] = (M @ nud_k) + (C @ nu_k) + (D @ nu_k) + gvec

    return tau_hat

# ============================================================
# === 4) Run predictions =====================================
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

pred = {}
residuals = {}
stats = {}
metrics = {}


i_dof = DOF_INDEX[ACTIVE_DOF]

for label in enabled_labels:
    p = PARAM_SETS[label]
    nu_pred_all = segmented_forward_prediction(
        t=t,
        nu_meas=nu_meas,
        eta_meas=eta_meas,
        tau=tau,
        params=p,
        segs=segs,
        include_ca=INCLUDE_CA,
        include_g=INCLUDE_G,
    )
    pred[label] = nu_pred_all[:, i_dof]
    residuals[label] = np.where(np.isfinite(pred[label]), nu_meas[:, i_dof] - pred[label], np.nan)
    stats[label] = residual_stats(residuals[label])
    metrics[label] = error_metrics(nu_meas[:, i_dof], pred[label])



# ============================================================
# === 4b) HMC posterior predictive band (segmented + filtered) =
# ============================================================
u_p05 = u_p50 = u_p95 = None
if ENABLE_HMC_BAND and (theta_plot is not None) and (param_names is not None):
    param_idx = {name: i for i, name in enumerate(param_names)}
    Ns = theta_plot.shape[0]
    Tn = len(t)

    U_list = []
    bad = 0
    bad_reasons = {"linAlg": 0, "nan": 0, "umax": 0, "nonfinite": 0}

    U_ABS_MAX = 10.0
    NAN_FRAC_MAX = 0.01

    if HMC_BASELINE == "MAP":
        base_params = params_map
    elif HMC_BASELINE == "SIM_TRUTH":
        base_params = params_sim_truth
    elif HMC_BASELINE == "SIM_MLE":
        base_params = params_sim_mle
    else:
        raise ValueError(f"Unknown HMC_BASELINE='{HMC_BASELINE}'")

    print(f"[HMC band] Using baseline: {HMC_BASELINE}")
    print(f"[HMC band] U_ABS_MAX={U_ABS_MAX}, NAN_FRAC_MAX={NAN_FRAC_MAX}")

    for j in range(Ns):
        p = dict(base_params)
        for name in HMC_BAND_PARAM_SUBSET:
            if (name in param_idx) and (name in p):
                p[name] = float(theta_plot[j, param_idx[name]])

        try:
            nu_pred_all = segmented_forward_prediction(
                t=t,
                nu_meas=nu_meas,
                eta_meas=eta_meas,
                tau=tau,
                params=p,
                segs=segs,
                include_ca=INCLUDE_CA,
                include_g=INCLUDE_G,
            )
        except np.linalg.LinAlgError:
            bad += 1
            bad_reasons["linAlg"] += 1
            continue

        u = nu_pred_all[:, i_dof].astype(float)

        # Evaluate only on predicted samples (finite)
        finite_u = u[np.isfinite(u)]
        if finite_u.size == 0:
            bad += 1
            bad_reasons["nonfinite"] += 1
            continue

        nan_frac = float(np.mean(~np.isfinite(u)))
        umax = float(np.nanmax(np.abs(finite_u)))

        reject = False
        if nan_frac > NAN_FRAC_MAX:
            reject = True
            bad_reasons["nan"] += 1
        if not np.isfinite(umax):
            reject = True
            bad_reasons["nonfinite"] += 1
        if umax > U_ABS_MAX:
            reject = True
            bad_reasons["umax"] += 1

        if reject:
            bad += 1
            continue

        U_list.append(u)

    if len(U_list) < 10:
        print(f"[HMC band] Too few valid trajectories: kept {len(U_list)}/{Ns}.")
        print("[HMC band] reject breakdown:", bad_reasons)
    else:
        U = np.vstack(U_list)  # (Nkeep, Tn)
        u_p05 = np.nanpercentile(U, 5, axis=0)
        u_p50 = np.nanpercentile(U, 50, axis=0)
        u_p95 = np.nanpercentile(U, 95, axis=0)

        w = u_p95 - u_p05
        print(f"[HMC band] kept {len(U_list)}/{Ns} (rejected {bad})")
        print("[HMC band] reject breakdown:", bad_reasons)
        print("band width stats:",
              "min", float(np.nanmin(w)),
              "median", float(np.nanmedian(w)),
              "max", float(np.nanmax(w)))

# ============================================================
# === 4c) Accel consistency check (masked, no integration) =====
# ============================================================
accel_pred = {}
accel_residuals = {}
accel_stats = {}

if PLOT_ACCEL_CHECK:
    accel_enabled = []
    if PLOT_ACCEL_MLE and PLOT_MLE: accel_enabled.append("MLE")
    if PLOT_ACCEL_TEST_1 and PLOT_TEST_PARAMS_1: accel_enabled.append("TEST_1")
    if PLOT_ACCEL_TEST_2 and PLOT_TEST_PARAMS_2: accel_enabled.append("TEST_2")
    if PLOT_ACCEL_MAP and PLOT_MAP: accel_enabled.append("MAP")
    if PLOT_ACCEL_HMC_TEST_1 and PLOT_HMC_TEST_PARAMS_1: accel_enabled.append("HMC_TEST_1")

    if PLOT_ACCEL_SIM:
        if PLOT_SIM_MLE: accel_enabled.append("SIM_MLE")
        if PLOT_SIM_TRUTH: accel_enabled.append("SIM_TRUTH")

    # mask: must be valid for state + have nu_dot
    accel_mask = valid_mask & np.isfinite(nu_dot_meas).all(axis=1)

    for label in accel_enabled:
        p = PARAM_SETS[label]
        nu_dot_hat = predict_nu_dot_from_csv_states_masked(
            nu_meas=nu_meas,
            eta_meas=eta_meas,
            tau=tau,
            params=p,
            include_ca=INCLUDE_CA,
            include_g=INCLUDE_G,
            mask=accel_mask
        )
        accel_pred[label] = nu_dot_hat[:, i_dof]
        accel_residuals[label] = np.where(np.isfinite(accel_pred[label]), nu_dot_meas[:, i_dof] - accel_pred[label], np.nan)
        accel_stats[label] = residual_stats(accel_residuals[label])

# ============================================================
# === 5) Plot: tau, velocity, residual =======================
# ============================================================
meta = DOF_META[ACTIVE_DOF]
nu_meas_dof = nu_meas[:, i_dof]
tau_dof = tau[:, i_dof]


TITLE_FS  = 20
AXIS_FS   = 18
TICK_FS   = 14
LEGEND_FS = 13

TAU_LW    = 2.6
MEAS_LW   = 3.6
PRED_LW   = 1.9
RESID_LW  = 2.0
MEAN_LW   = 1.6

# --- consistent colors per method ---
METHOD_COLORS = {
    "MLE": "#ff7f0e",       # orange
    "MAP": "#2ca02c",       # green
    "SIM_TRUTH": "#d62728", # red
    "SIM_MLE": "#9467bd",   # purple
    "TEST_1": "#8c564b",    # brown
    "TEST_2": "#e377c2",    # pink
    "HMC_TEST_1": "#7f7f7f" # gray
}

def color_for(label: str):
    # fallback if you add new labels later
    return METHOD_COLORS.get(label, None)


plt.figure(figsize=(14, 10))

# 1) tau
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau_dof, linewidth=TAU_LW, label=f"{meta['tau']} applied")
ax1.set_ylabel(f"{meta['tau']} [{meta['tau_unit']}]", fontsize=AXIS_FS, fontweight="bold")
ax1.grid(True, alpha=0.35)
ax1.tick_params(axis="both", labelsize=TICK_FS)
ax1.legend(fontsize=LEGEND_FS, framealpha=0.95)

# 2) velocity
ax2 = plt.subplot(3, 1, 2, sharex=ax1)

# measured backbone (bold blue)
ax2.plot(
    t_rel, nu_meas_dof,
    linewidth=MEAS_LW,
    color="#1f77b4",
    label=f"{meta['sym']} measured",
    zorder=4
)

# predicted overlays (orange on top)
for label in enabled_labels:
    ax2.plot(
        t_rel, pred[label],
        linewidth=PRED_LW,
        color=color_for(label),
        alpha=0.95,
        label=f"{meta['sym']} predicted ({label})",
        zorder=6
    )




# HMC band
if ENABLE_HMC_BAND and (u_p05 is not None):
    ax2.fill_between(
        t_rel, u_p05, u_p95,
        color="#c9a300",
        alpha=0.40,
        zorder=1,
        label="HMC 90% band"
    )

    ax2.plot(
        t_rel, u_p50,
        linestyle="--",
        linewidth=2.6,
        color="red",
        zorder=7,
        label="HMC median"
    )

ax2.set_ylabel(f"{meta['sym']} [{meta['unit']}]", fontsize=AXIS_FS, fontweight="bold")
ax2.set_ylim(list(meta["ylim"]))
ax2.grid(True, alpha=0.35)
ax2.tick_params(axis="both", labelsize=TICK_FS)
ax2.legend(
    loc="best",
    fontsize=11,
    framealpha=0.95,
    ncol=2,
    handlelength=1.2,
    columnspacing=0.8,
    labelspacing=0.3,
    borderpad=0.3
)

# 3) residuals
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
for label in enabled_labels:
    mu, rmse = stats[label]
    m = metrics[label]

    stat_txt = (
        f"mean={mu:+.4f} {meta['unit']}, "
        f"rmse={m['rmse']:.4f} {meta['unit']}, "
        f"nrmseσ={m['nrmse_sigma']:.3f}, "
        f"%rmse={m['prmse']:.1f}%"
    )


    ax3.plot(
        t_rel,
        residuals[label],
        linewidth=RESID_LW,
        color=color_for(label),
        label=f"Residual ({label})  [{stat_txt}]"
    )

    if label == enabled_labels[0] and np.isfinite(mu):
        ax3.axhline(
            mu,
            linestyle="--",
            linewidth=MEAN_LW,
            color="0.3",
            label="Mean residual"
        )


ax3.set_ylabel("Residual", fontsize=AXIS_FS, fontweight="bold")
ax3.set_xlabel("Time [s]", fontsize=AXIS_FS, fontweight="bold")
ax3.grid(True, alpha=0.35)
ax3.tick_params(axis="both", labelsize=TICK_FS)
ax3.legend(fontsize=LEGEND_FS, framealpha=0.95)

plt.suptitle(f"{meta['name']} ({ACTIVE_DOF}) Post-Check", fontsize=TITLE_FS, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.show()

# ============================================================
# === 5c) Plot accel consistency check (optional) =============
# ============================================================
if PLOT_ACCEL_CHECK and len(accel_pred) > 0:
    toggle_line = f"Includes C_RB{' + C_A' if INCLUDE_CA else ''}{', includes g(eta)' if INCLUDE_G else ', excludes g(eta)'}"
    ameta = ACCEL_META[ACTIVE_DOF]
    nu_dot_meas_dof = nu_dot_meas[:, i_dof]

    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(t_rel, nu_dot_meas_dof, linewidth=2, label=f"{ameta['sym']} truth (CSV)")
    for label in accel_pred.keys():
        ax1.plot(t_rel, accel_pred[label], linewidth=2, label=f"{ameta['sym']} predicted ({label})")

    ax1.set_ylabel(f"{ameta['sym']} [{ameta['unit']}]")
    ax1.set_title(f"{ameta['name']} accel consistency check (MASKED): CSV ν,η,τ → predicted ν̇\n{toggle_line}")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    for label in accel_pred.keys():
        mu, rmse = accel_stats[label]
        ax2.plot(t_rel, accel_residuals[label], label=f"Residual ({label})  mean={mu:+.4e}, rmse={rmse:.4e}")
        if np.isfinite(mu):
            ax2.axhline(mu, linestyle="--", linewidth=1)

    ax2.set_ylabel(f"Residual [{ameta['unit']}]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# === Amplitude-dependent residual metrics (by |tau| bins) ====
# ============================================================
abs_tau = np.abs(tau_dof)

# Define bins (tweak these if you want)
bins = [
    ("|τ| ≤ 25", abs_tau <= 25),
    ("25 < |τ| ≤ 50", (abs_tau > 25) & (abs_tau <= 50)),
    ("|τ| > 50", abs_tau > 50),
]

print("\n=== Amplitude-dependent error metrics (by |tau|) ===")
for label in enabled_labels:
    print(f"\n[{label}]")
    for name, msk in bins:
        mm = error_metrics_masked(nu_meas_dof, pred[label], msk)
        print(
            f"  {name:12s}  N={mm['n']:5d}  "
            f"RMSE={mm['rmse']:.4f} {meta['unit']}  "
            f"NRMSEσ={mm['nrmse_sigma']:.3f}  "
            f"%RMSE={mm['prmse']:.1f}%"
        )
print("====================================================\n")


# ============================================================
# === 5d) Plot tau residual diagnostics (optional) ============
# ============================================================
if PLOT_TAU_RESIDUAL_DIAGNOSTICS:
    labels_to_plot = [lab for lab in TAU_RESIDUAL_LABELS if lab in PARAM_SETS]
    if len(labels_to_plot) == 0:
        print("[tau residual diagnostics] No valid labels in TAU_RESIDUAL_LABELS.")
    else:
        meta = DOF_META[ACTIVE_DOF]

        TITLE_FS = 18
        AXIS_FS  = 16
        TICK_FS  = 13
        LEGEND_FS = 13

        CMD_LW = 2.8
        MODEL_LW = 2.2
        RESID_LW = 1.6
        THR_LW = 2.0

        plt.figure(figsize=(14, 8))
        axL = plt.subplot(1, 1, 1)
        axR = axL.twinx()

        tau_cmd = tau[:, i_dof].astype(float)

        # mask: only compute inverse dynamics when nu, nu_dot, eta are finite
        tau_mask = (
            np.isfinite(nu_meas).all(axis=1) &
            np.isfinite(nu_dot_meas).all(axis=1) &
            np.isfinite(eta_meas).all(axis=1) &
            np.isfinite(tau_cmd)
        )

        for label in labels_to_plot:
            p = PARAM_SETS[label]

            tau_hat = predict_tau_from_csv_states_masked(
                nu_meas=nu_meas,
                nu_dot_meas=nu_dot_meas,
                eta_meas=eta_meas,
                params=p,
                include_ca=INCLUDE_CA,
                include_g=INCLUDE_G,
                mask=tau_mask
            )[:, i_dof].astype(float)

            r_abs = np.abs(tau_cmd - tau_hat)
            rr = r_abs[np.isfinite(r_abs)]
            if rr.size == 0:
                print(f"[tau residual diagnostics] No finite samples for {label}.")
                continue

            thr = np.nanpercentile(rr, TAU_RESIDUAL_PCT)

            axL.plot(t_rel, tau_hat, linewidth=MODEL_LW, label="Model X", zorder=2)
            axL.plot(t_rel, tau_cmd, linewidth=CMD_LW, label="Commanded X", zorder=5)

            axR.plot(t_rel, r_abs, linewidth=RESID_LW, alpha=0.95, label="Residual", zorder=3)
            axR.axhline(thr, linestyle="--", linewidth=THR_LW, color="red", alpha=0.95, label=f"{TAU_RESIDUAL_PCT}th percentile", zorder=4)

            mask = np.isfinite(r_abs) & (r_abs > thr)
            if np.any(mask):
                idx = np.where(mask)[0]
                breaks = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[idx[0], idx[breaks + 1]]
                ends   = np.r_[idx[breaks], idx[-1]]
                for s, e in zip(starts, ends):
                    axL.axvspan(t_rel[s], t_rel[e], alpha=0.12, zorder=1)

        axL.set_title(f"{meta['name']} ({ACTIVE_DOF}) τ Residual Diagnostics", fontsize=TITLE_FS)
        axL.set_xlabel("Time [s]", fontsize=AXIS_FS)
        axL.set_ylabel(f"{meta['tau']} [N]" if meta["tau_unit"] == "N" else f"{meta['tau']} [{meta['tau_unit']}]", fontsize=AXIS_FS)
        axR.set_ylabel("Residual Magnitude", fontsize=AXIS_FS)

        axL.tick_params(axis="both", labelsize=TICK_FS)
        axR.tick_params(axis="y", labelsize=TICK_FS)
        axL.grid(True, alpha=0.35)
        axR.grid(False)

        h1, l1 = axL.get_legend_handles_labels()
        h2, l2 = axR.get_legend_handles_labels()

        wanted = ["Model X", "Commanded X", "Residual", f"{TAU_RESIDUAL_PCT}th percentile"]
        combined = list(zip(h1 + h2, l1 + l2))

        filtered = []
        seen = set()
        for name in wanted:
            for h, l in combined:
                if l == name and l not in seen:
                    filtered.append((h, l))
                    seen.add(l)

        if filtered:
            handles, labels = zip(*filtered)
            axL.legend(handles, labels, loc="upper left", fontsize=LEGEND_FS, framealpha=0.95)

        plt.tight_layout()
        plt.show()
# ============================================================
# === Plot Truth (MOCAP) XY Track – First 30 Seconds =========
# ============================================================

T_WINDOW = 30.0  # seconds

# Time mask
time_mask = t_rel <= T_WINDOW

x_all = eta_meas[:, 0].astype(float)
y_all = eta_meas[:, 1].astype(float)

# Build mask
mask = (
    np.isfinite(x_all) &
    np.isfinite(y_all) &
    time_mask
)

if pose_valid is not None:
    mask &= (pose_valid > 0.5)

# Create NaN-separated arrays
x_plot = np.full_like(x_all, np.nan)
y_plot = np.full_like(y_all, np.nan)
x_plot[mask] = x_all[mask]
y_plot[mask] = y_all[mask]

# Shift to start at origin
valid_indices = np.where(np.isfinite(x_plot) & np.isfinite(y_plot))[0]
if len(valid_indices) > 0:
    i0 = valid_indices[0]
    x_plot -= x_plot[i0]
    y_plot -= y_plot[i0]

plt.figure(figsize=(8, 8))
plt.plot(x_plot, y_plot, linewidth=2.0)
plt.xlabel("X Position [m]", fontsize=14)
plt.ylabel("Y Position [m]", fontsize=14)
plt.title("Defender Teleop Circular Maneuver (First 30 s)", fontsize=16)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# ============================================================
# === 6) dt sanity print =====================================
# ============================================================
dt_vec = np.diff(t)
print("dt stats:", np.nanmin(dt_vec), np.nanmean(dt_vec), np.nanmax(dt_vec))
