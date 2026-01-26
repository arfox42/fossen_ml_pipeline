#!/usr/bin/env python3
"""
post_check_full6dof_single_dof_segmented.py
-------------------------------------------
Forward-integrates full 6DOF Fossen dynamics, but plots ONLY the selected ACTIVE_DOF
channel (velocity + residual), plus an optional accel consistency check.

KEY FEATURE:
- CSV may contain NaN/Inf due to mocap dropout.
- We DO NOT rewrite/clean the CSV.
- We run "segmented integration":
    * integrate only across contiguous valid segments
    * stop at invalid region, restart at next valid sample (new initial condition)
- Residual mean/RMSE computed ONLY over valid predicted samples.

TWO MLE PARAMETER LIBRARIES (user-requested):
- MLE_BASE     : your original MLE parameter set (no-dropout params)
- MLE_DROPOUT  : MLE parameters learned after accounting for mocap dropout

CSV HEADER EXPECTED (31 columns):
time
u_dot v_dot w_dot p_dot q_dot r_dot
u v w p q r
x y z phi theta psi
X Y Z K M N
pose_valid pose_age twist_valid twist_age wrench_valid wrench_age
"""

import numpy as np
import matplotlib.pyplot as plt
import torch  # only needed if you later enable HMC band loading


# ============================================================
# === 0) Config / Flags ======================================
# ============================================================

# --- Dataset / DOF ---
ACTIVE_DOF = "Y"   # "X","Y","Z","K","M","N"
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Y_Data/Tank_data/defender_data_y_run_mocap_dropout.csv"

# --- Validity gating (recommended ON for segmented integration) ---
USE_VALIDITY_FLAGS = True
REQUIRE_POSE_VALID = False   # usually not needed for single-DOF velocity plots
REQUIRE_TWIST_VALID = True   # u,v,w,p,q,r
REQUIRE_WRENCH_VALID = True  # X,Y,Z,K,M,N

# Optional: "freshness" gating (seconds). None disables.
MAX_ALLOWED_AGE_S = None     # e.g., 0.05

# Minimum segment length to integrate (samples)
MIN_SEG_LEN = 50  # e.g., 50 @100Hz = 0.5s

# --- Physics toggles ---
INCLUDE_CA = True   # True -> use C_RB + C_A, False -> use C_RB only
INCLUDE_G  = True   # include restoring g(eta)

# --- Integrator timing ---
FORCE_FIXED_DT = True
DT_FIXED = 0.01

# ============================================================
# === Plot toggles (THIS IS THE PART YOU WANTED) =============
# ============================================================

# --- Two MLE libraries ---
PLOT_MLE_BASE    = True    # original MLE params (no-dropout)
PLOT_MLE_DROPOUT = False    # NEW MLE params learned with dropout-aware CSV

# --- Optional additional overlays ---
PLOT_MAP     = True
PLOT_HMC_MAP = False

# --- Posterior predictive band (HMC samples) ---
ENABLE_HMC_BAND = False
HMC_SAMPLES_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_yaw_circle_samples.pt"
HMC_BAND_NPLOT = 400

# --- Optional accel check (no integration) ---
PLOT_ACCEL_CHECK       = False
PLOT_ACCEL_MLE_BASE    = True
PLOT_ACCEL_MLE_DROPOUT = False
PLOT_ACCEL_MAP         = True
PLOT_ACCEL_HMC_MAP     = False


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
# === 1) Load CSV (31 columns) ================================
# ============================================================
data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)

t = data[:, 0].astype(float)
t_rel = t - t[0]

# ν_dot
nu_dot_meas = np.vstack([
    data[:, 1].astype(float),
    data[:, 2].astype(float),
    data[:, 3].astype(float),
    data[:, 4].astype(float),
    data[:, 5].astype(float),
    data[:, 6].astype(float),
]).T

# ν
nu_meas = np.vstack([
    data[:, 7].astype(float),
    data[:, 8].astype(float),
    data[:, 9].astype(float),
    data[:, 10].astype(float),
    data[:, 11].astype(float),
    data[:, 12].astype(float),
]).T

# η
eta_meas = np.vstack([
    data[:, 13].astype(float),
    data[:, 14].astype(float),
    data[:, 15].astype(float),
    data[:, 16].astype(float),
    data[:, 17].astype(float),
    data[:, 18].astype(float),
]).T

# τ
tau = np.vstack([
    data[:, 19].astype(float),
    data[:, 20].astype(float),
    data[:, 21].astype(float),
    data[:, 22].astype(float),
    data[:, 23].astype(float),
    data[:, 24].astype(float),
]).T

# validity / age
pose_valid   = data[:, 25].astype(float)
pose_age     = data[:, 26].astype(float)
twist_valid  = data[:, 27].astype(float)
twist_age    = data[:, 28].astype(float)
wrench_valid = data[:, 29].astype(float)
wrench_age   = data[:, 30].astype(float)


# ============================================================
# === 1b) Valid mask + segments ===============================
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
        b = int(idx[e] + 1)  # slice end
        if (b - a) >= min_len:
            segs_out.append((a, b))
    return segs_out

def build_valid_mask():
    # must have finite nu/eta/tau for integration
    m = (
        np.isfinite(nu_meas).all(axis=1) &
        np.isfinite(eta_meas).all(axis=1) &
        np.isfinite(tau).all(axis=1)
    )

    if USE_VALIDITY_FLAGS:
        if REQUIRE_POSE_VALID:
            m &= (pose_valid > 0.5)
        if REQUIRE_TWIST_VALID:
            m &= (twist_valid > 0.5)
        if REQUIRE_WRENCH_VALID:
            m &= (wrench_valid > 0.5)

        if MAX_ALLOWED_AGE_S is not None:
            if REQUIRE_POSE_VALID:
                m &= (pose_age <= MAX_ALLOWED_AGE_S)
            if REQUIRE_TWIST_VALID:
                m &= (twist_age <= MAX_ALLOWED_AGE_S)
            if REQUIRE_WRENCH_VALID:
                m &= (wrench_age <= MAX_ALLOWED_AGE_S)

    return m

valid_mask = build_valid_mask()
segs = contiguous_segments(valid_mask, min_len=MIN_SEG_LEN)

def pct(x): return 100.0 * float(np.mean(x > 0.5))
print("\n=== Validity stats from CSV ===")
print(f"pose_valid  : {pct(pose_valid):.2f}% valid")
print(f"twist_valid : {pct(twist_valid):.2f}% valid")
print(f"wrench_valid: {pct(wrench_valid):.2f}% valid")
print("================================\n")
print(f"[INFO] Found {len(segs)} valid integration segments (min_len={MIN_SEG_LEN} samples).")

if len(segs) == 0:
    raise RuntimeError("No valid segments found. Relax validity settings or check CSV.")


# ============================================================
# === 2) Parameters ==========================================
# ============================================================
G = 9.8
params_base = {"W": 23.89 * G}

# -------------------------
# MLE_BASE (NO-DROPOUT params)  <-- your original set
# -------------------------
params_mle_base = {
    **params_base,

    "m": 23.89,
    "B": 236.00,
    "I_xx": 0.5,
    "I_yy": 1.76,
    "I_zz": 2.13,

    # CG / CB (m)
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,

    # Added mass
    "X_dot_u": -33.61,
    "Y_dot_v": -31.56,
    "Z_dot_w": -79.58,
    "K_dot_p": -0.1,
    "M_dot_q": -0.46,
    "N_dot_r": -0.70,

    # Linear & quadratic damping
    "X_u": -16.49, "X_uu": -42.49,
    "Y_v": -34.05, "Y_vv": -108.74,
    "Z_w": -35.66, "Z_ww": -128.31,
    "K_p": -1.24, "K_pp": -0.08,
    "M_q": -2.08, "M_qq": -1.61,
    "N_r": -2.88, "N_rr": -2.69,
}

# -------------------------
# MLE_DROPOUT (WITH-DROPOUT params) <-- your new set
# -------------------------
params_mle_dropout = {
    **params_base,

    # Physical
    "m": 23.89,
    "B": 232.04,
    "I_xx": 0.5,
    "I_yy": 1.76,
    "I_zz": 2.13,

    # CG / CB (m)
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,

    # Added mass
    "X_dot_u": -33.61,
    "Y_dot_v": -11.48,
    "Z_dot_w": -56.81,
    "K_dot_p": -0.1,
    "M_dot_q": -0.46,
    "N_dot_r": -0.70,

    # Linear & quadratic damping
    "X_u": -16.49, "X_uu": -42.49,
    "Y_v": -27.27, "Y_vv": -107.24,
    "Z_w": -38.84, "Z_ww": -124.92,
    "K_p": -1.24, "K_pp": -0.08,
    "M_q": -2.08, "M_qq": -1.61,
    "N_r": -2.88, "N_rr": -2.69,
}

# Optional MAP/HMC (unchanged placeholders)
params_map = {
    **params_base,

    "m": 23.89,
    "B": 239.81,
    "I_xx": 0.387,
    "I_yy": 1.28,
    "I_zz": 1.42,

    "x_cg": 0.0, "y_cg": 0.0, "z_cg": -0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,

    "X_dot_u": -32.41,
    "Y_dot_v": -16.78,
    "Z_dot_w": -54.02,
    "K_dot_p": -0.079,
    "M_dot_q": -0.26,
    "N_dot_r": -0.29,

    "X_u":  -1.01,    "X_uu": -62.05,
    "Y_v":  -0.93,   "Y_vv": -137.19,
    "Z_w":  -35.79,  "Z_ww": -128.99,
    "K_p":  -1.5,    "K_pp": -2.1,
    "M_q":  -2.9,    "M_qq": -14.6,
    "N_r": -2.83,    "N_rr": -2.71,




}

params_hmc_map = {**params_base, "m": 17.2, "B": 168.56, "I_xx": 1.0, "I_yy": 1.0, "I_zz": 1.0,
                  "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0, "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,
                  "Y_dot_v": -22.584, "Z_dot_w": -22.3775, "K_dot_p": -0.079, "M_dot_q": -0.26, "N_dot_r": -0.286,
                  "Y_v": -12.6, "Y_vv": -102.006, "Z_w": -14.17, "Z_ww": -155.8358,
                  "K_p": -1.5, "K_pp": -2.1, "M_q": -2.9, "M_qq": -14.6, "N_r": -10.343, "N_rr": -8.8,
                  "X_dot_u": -18.0, "X_u": -4.66, "X_uu": -51.5}

PARAM_SETS = {
    "MLE_BASE": params_mle_base,
    "MLE_DROPOUT": params_mle_dropout,
    "MAP": params_map,
    "HMC_MAP": params_hmc_map,
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
        float(params["X_u"]),
        float(params["Y_v"]),
        float(params["Z_w"]),
        float(params["K_p"]),
        float(params["M_q"]),
        float(params["N_r"]),
    ], dtype=float)

    quad = np.array([
        float(params["X_uu"]),
        float(params["Y_vv"]),
        float(params["Z_ww"]),
        float(params["K_pp"]),
        float(params["M_qq"]),
        float(params["N_rr"]),
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

    x_G = float(params["x_cg"])
    y_G = float(params["y_cg"])
    z_G = float(params["z_cg"])

    x_B = float(params["x_cb"])
    y_B = float(params["y_cb"])
    z_B = float(params["z_cb"])

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

def forward_integrate_6dof_segment(t_seg, nu0, eta0, tau_seg, params, include_ca=False, include_g=False):
    N = len(t_seg)
    nu_pred = np.zeros((N, 6), dtype=float)
    eta_pred = np.zeros((N, 6), dtype=float)
    nu_dot_pred = np.zeros((N, 6), dtype=float)

    nu_pred[0, :] = np.asarray(nu0, dtype=float).reshape(6,)
    eta_pred[0, :] = np.asarray(eta0, dtype=float).reshape(6,)

    M = M_total(params)

    for k in range(1, N):
        dt = DT_FIXED if FORCE_FIXED_DT else float(t_seg[k] - t_seg[k - 1])

        if (not np.isfinite(dt)) or (dt <= 0.0):
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

def residual_stats(residual: np.ndarray):
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return np.nan, np.nan
    mu = float(np.mean(r))
    rmse = float(np.sqrt(np.mean(r**2)))
    return mu, rmse

def predict_nu_dot_from_csv_states_masked(nu_meas, eta_meas, tau, params, include_ca=False, include_g=False, mask=None):
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


# ============================================================
# === 4) Run segmented predictions for enabled sets ===========
# ============================================================
enabled_labels = []
if PLOT_MLE_BASE:
    enabled_labels.append("MLE_BASE")
if PLOT_MLE_DROPOUT:
    enabled_labels.append("MLE_DROPOUT")
if PLOT_MAP:
    enabled_labels.append("MAP")
if PLOT_HMC_MAP:
    enabled_labels.append("HMC_MAP")

if len(enabled_labels) == 0:
    raise ValueError("No prediction methods enabled (turn on PLOT_MLE_BASE and/or PLOT_MLE_DROPOUT).")

pred_full = {}
residuals_full = {}
stats = {}

for label in enabled_labels:
    p = PARAM_SETS[label]

    nu_pred_all = np.full_like(nu_meas, np.nan, dtype=float)

    for (a, b) in segs:
        nu_pred_seg, _, _ = forward_integrate_6dof_segment(
            t_seg=t[a:b],
            nu0=nu_meas[a, :],
            eta0=eta_meas[a, :],
            tau_seg=tau[a:b, :],
            params=p,
            include_ca=INCLUDE_CA,
            include_g=INCLUDE_G,
        )
        nu_pred_all[a:b, :] = nu_pred_seg

    i = DOF_INDEX[ACTIVE_DOF]
    pred_full[label] = nu_pred_all[:, i]
    residuals_full[label] = nu_meas[:, i] - pred_full[label]
    stats[label] = residual_stats(residuals_full[label])


# ============================================================
# === 4c) Accel consistency check (masked, no integration) =====
# ============================================================
accel_pred = {}
accel_residuals = {}
accel_stats = {}

if PLOT_ACCEL_CHECK:
    accel_enabled = []
    if PLOT_ACCEL_MLE_BASE and PLOT_MLE_BASE:
        accel_enabled.append("MLE_BASE")
    if PLOT_ACCEL_MLE_DROPOUT and PLOT_MLE_DROPOUT:
        accel_enabled.append("MLE_DROPOUT")
    if PLOT_ACCEL_MAP and PLOT_MAP:
        accel_enabled.append("MAP")
    if PLOT_ACCEL_HMC_MAP and PLOT_HMC_MAP:
        accel_enabled.append("HMC_MAP")

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
        i = DOF_INDEX[ACTIVE_DOF]
        accel_pred[label] = nu_dot_hat[:, i]
        accel_residuals[label] = nu_dot_meas[:, i] - accel_pred[label]
        accel_stats[label] = residual_stats(accel_residuals[label])


# ============================================================
# === 5) Plot: tau, velocity, residual =======================
# ============================================================
meta = DOF_META[ACTIVE_DOF]
i_dof = DOF_INDEX[ACTIVE_DOF]

nu_meas_dof = nu_meas[:, i_dof]
tau_dof = tau[:, i_dof]

toggle_line = f"Includes C_RB{' + C_A' if INCLUDE_CA else ''}{', includes g(eta)' if INCLUDE_G else ', excludes g(eta)'}"

plt.figure(figsize=(14, 10))

# 1) tau
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau_dof, "r-", label=f"{meta['tau']} applied τ")
ax1.set_ylabel(f"{meta['tau']} [{meta['tau_unit']}]")
ax1.set_title(
    f"{meta['name']} ({ACTIVE_DOF}) Post-Check (SEGMENTED): measured {meta['sym']} vs predicted {meta['sym']}\n{toggle_line}"
)
ax1.grid(True)
ax1.legend()

# 2) velocity
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t_rel, nu_meas_dof, linewidth=2, label=f"{meta['sym']} measured")
for label in enabled_labels:
    ax2.plot(t_rel, pred_full[label], linewidth=2, label=f"{meta['sym']} predicted ({label})")
ax2.set_ylabel(f"{meta['sym']} [{meta['unit']}]")
ax2.set_ylim(list(meta["ylim"]))
ax2.grid(True)
ax2.legend()

# 3) residuals
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
for label in enabled_labels:
    mu, rmse = stats[label]
    ax3.plot(t_rel, residuals_full[label], label=f"Residual ({label})  mean={mu:+.4f}, rmse={rmse:.4f}")
    if np.isfinite(mu):
        ax3.axhline(mu, linestyle="--", linewidth=1)
ax3.set_ylabel(f"Residual [{meta['unit']}]")
ax3.set_xlabel("Time [s]")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()


# ============================================================
# === 5c) Plot accel consistency check (optional) =============
# ============================================================
if PLOT_ACCEL_CHECK and len(accel_pred) > 0:
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
# === 6) sanity prints =======================================
# ============================================================
dt_vec = np.diff(t)
print("dt stats:", np.min(dt_vec), np.mean(dt_vec), np.max(dt_vec))

total_valid = int(np.sum(valid_mask))
print(f"[INFO] valid_mask True samples: {total_valid} / {len(valid_mask)} ({100*total_valid/len(valid_mask):.2f}%)")
for k, (a, b) in enumerate(segs[:10]):
    print(f"[INFO] seg {k:02d}: rows [{a}:{b})  len={b-a}  t_rel=[{t_rel[a]:.2f}, {t_rel[b-1]:.2f}] s")
if len(segs) > 10:
    print(f"[INFO] ... {len(segs)-10} more segments")
