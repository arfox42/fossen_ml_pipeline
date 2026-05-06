#!/usr/bin/env python3
"""
surge_post_check_fullX.py
-------------------------------------
Forward-integrates SURGE (u) only for the Defender, using all terms that affect X:

    (m - X_dot_u) * u_dot = X_applied - (C_RB(nu) nu)_X - X_drag(u) - g_X

Hybrid forward sim:
- Integrate ONLY u(t)
- Compute rigid-body Coriolis coupling term (C_RB nu)_X using MEASURED v,w,q,r
- DO NOT include C_A per your sim setting
samples.
Overlays three predicted traces on measured u:
  1) MLE
  2) MAP
  3) HMC "MAP-from-samples"
rov_numerics = rov.drop(columns=["time", rov.columns[-2], rov.columns[-1]])
Adds residual mean (and RMSE) and shows them on the residual plot + in legend.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# === 0) Config / Flags ======================================
# ============================================================
INCLUDE_CA = False   # True -> use C_RB + C_A, False -> use C_RB only
INCLUDE_G  = True   # optional: restoring (set True later if needed)
FORCE_FIXED_DT = False
DT_FIXED = 0.1
ACTIVE_DOF = "X"   # options: "X","Y","Z","K","M","N","ALL"

DOF_INDEX = {
    "X": 0,  # u
    "Y": 1,  # v
    "Z": 2,  # w
    "K": 3,  # p
    "M": 4,  # q
    "N": 5,  # r
}

# ============================================================
# === 1) Load CSV ============================================
# ============================================================
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/X_Data/Sim_data/csv_full_truth_x_run_26OCT.csv"

data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)

t = data[:, 0].astype(float)
t_rel = t - t[0]

# Measured body velocities ν = [u v w p q r]
u_meas = data[:, 7].astype(float)
v_meas = data[:, 8].astype(float)
w_meas = data[:, 9].astype(float)
p_meas = data[:, 10].astype(float)
q_meas = data[:, 11].astype(float)
r_meas = data[:, 12].astype(float)

# Body velocity vector ν = [u v w p q r]
nu_meas = np.vstack((
    u_meas,
    v_meas,
    w_meas,
    p_meas,
    q_meas,
    r_meas
)).T   # shape (N, 6)


# Applied forces/moments τ = [X Y Z K M N]
tau_X = data[:, 19].astype(float)
tau_Y = data[:, 20].astype(float)
tau_Z = data[:, 21].astype(float)
tau_K = data[:, 22].astype(float)
tau_M = data[:, 23].astype(float)
tau_N = data[:, 24].astype(float)

tau = np.vstack((tau_X, tau_Y, tau_Z, tau_K, tau_M, tau_N)).T  # shape (N,6)

# Body accelerations ν_dot = [u_dot v_dot w_dot p_dot q_dot r_dot]
# (from truth sim / IMU topic in CSV)
u_dot_meas = data[:, 1].astype(float)
v_dot_meas = data[:, 2].astype(float)
w_dot_meas = data[:, 3].astype(float)
p_dot_meas = data[:, 4].astype(float)
q_dot_meas = data[:, 5].astype(float)
r_dot_meas = data[:, 6].astype(float)

nu_dot_meas = np.vstack((
    u_dot_meas,
    v_dot_meas,
    w_dot_meas,
    p_dot_meas,
    q_dot_meas,
    r_dot_meas
)).T   # shape (N, 6)

# Position / attitude η = [x y z phi theta psi]
x_meas     = data[:, 13].astype(float)
y_meas     = data[:, 14].astype(float)
z_meas     = data[:, 15].astype(float)
phi_meas   = data[:, 16].astype(float)
theta_meas = data[:, 17].astype(float)
psi_meas   = data[:, 18].astype(float)

eta_meas = np.vstack((
    x_meas,
    y_meas,
    z_meas,
    phi_meas,
    theta_meas,
    psi_meas
)).T   # shape (N, 6)


# ============================================================
# === 2) Parameters (base + full per-method dicts) ============
# ============================================================

G = 9.81  # m/s^2

# Base parameters: weight only (per your request)
params_base = {
    "W": 17.2 * G,   # [N]
}

# -------------------------
# MLE parameters (full 6DOF)
# -------------------------
params_mle = {
    **params_base,

    # Physical
    "m": 17.2,
    "B": 168.56,          # buoyancy [N]
    "I_xx": 1.0,
    "I_yy": 1.0,
    "I_zz": 1.0,

    # CG / CB (m)
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,

    # Added mass (non-surge from ROS node)
    # (surge X_dot_u provided below)
    "Y_dot_v": -22.584,
    "Z_dot_w": -22.3775,
    "K_dot_p": -0.079,
    "M_dot_q": -0.26,
    "N_dot_r": -0.6333212852478027,

    # Linear & quadratic damping (non-surge from ROS node)
    # (surge X_u, X_uu provided below)
    "Y_v":  -12.6,     "Y_vv": -102.006,
    "Z_w":  -14.17,    "Z_ww": -155.8358,
    "K_p":  -1.5,      "K_pp": -2.1,
    "M_q":  -2.9,      "M_qq": -14.6,
    "N_r":  -10.377045631408691,   "N_rr": -8.736456871032715,

    # Surge terms (MLE-learned)
    "X_dot_u": -4.66,
    "X_u":     -4.717586517333984,
    "X_uu":    -51.50349197387695,
}


# -------------------------
# MAP parameters (full 6DOF)
# -------------------------
params_map = {
    **params_base,

    # Physical
    "m": 17.2,
    "B": 168.56,          # buoyancy [N]
    "I_xx": 1.0,
    "I_yy": 1.0,
    "I_zz": 1.0,

    # CG / CB (m)
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,

    # Added mass (non-surge from ROS node)
    "Y_dot_v": -22.584,
    "Z_dot_w": -22.3775,
    "K_dot_p": -0.079,
    "M_dot_q": -0.26,
    "N_dot_r": -0.273586,

    # Linear & quadratic damping (non-surge from ROS node)
    "Y_v":  -8.25,     "Y_vv": -102.006,
    "Z_w":  -14.17,    "Z_ww": -155.8358,
    "K_p":  -1.5,      "K_pp": -2.1,
    "M_q":  -2.9,      "M_qq": -14.6,
    "N_r":  -10.343,   "N_rr": -8.8,

    # Surge terms (MAP-learned)
    "X_dot_u": -18.0,
    "X_u":     -4.66,
    "X_uu":    -51.5,
}

# -----------------------------
# HMC_MAP parameters (full 6DOF)
# -----------------------------
params_hmc_map = {
    **params_base,

    # Physical
    "m": 17.2,
    "B": 168.56,          # buoyancy [N]
    "I_xx": 1.0,
    "I_yy": 1.0,
    "I_zz": 1.0,

    # CG / CB (m)
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.05,

    # Added mass (non-surge from ROS node)
    "Y_dot_v": -22.584,
    "Z_dot_w": -22.3775,
    "K_dot_p": -0.079,
    "M_dot_q": -0.26,
    "N_dot_r": -0.286,

    # Linear & quadratic damping (non-surge from ROS node)
    "Y_v":  -12.6,     "Y_vv": -102.006,
    "Z_w":  -14.17,    "Z_ww": -155.8358,
    "K_p":  -1.5,      "K_pp": -2.1,
    "M_q":  -2.9,      "M_qq": -14.6,
    "N_r":  -10.343,   "N_rr": -8.8,

    # Surge terms (HMC MAP point)
    "X_dot_u": -18.0,
    "X_u":     -4.66,
    "X_uu":    -51.5,
}

PARAM_SETS = {
    "MLE": params_mle,
    "MAP": params_map,
    "HMC_MAP": params_hmc_map,
}

# ============================================================
# === 2b) Load HMC samples for predictive band ===============
# ============================================================
HMC_SAMPLES_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_yaw_circle_samples.pt"

hmc_data = torch.load(HMC_SAMPLES_PATH, map_location="cpu")
theta_samples = hmc_data["samples"].detach().cpu().numpy()   # (N, 26)
param_names   = list(hmc_data["param_names"])

# Build name -> column index map for all parameters
idx = {name: i for i, name in enumerate(param_names)}

# Sanity check that we have everything we expect
required = [
    "X_dot_u","Y_dot_v","Z_dot_w","K_dot_p","M_dot_q","N_dot_r",
    "I_xx","I_yy","I_zz",
    "x_g","y_g","z_g",
    "X_u","Y_v","Z_w","K_p","M_q","N_r",
    "X_uu","Y_vv","Z_ww","K_pp","M_qq","N_rr",
    "B","z_b"
]
missing = [k for k in required if k not in idx]
if missing:
    raise ValueError(f"HMC sample file missing required params: {missing}")

# Optional: thin / subsample for speed (you will likely reduce this for full 6DOF)
N_plot = 400  # start here once full 6DOF is running; 2000 will be slow
if theta_samples.shape[0] > N_plot:
    sel = np.random.choice(theta_samples.shape[0], size=N_plot, replace=False)
    theta_plot = theta_samples[sel]
else:
    theta_plot = theta_samples


def sample_to_params(theta_row: np.ndarray, m_fixed: float = 17.2) -> dict:
    """
    Convert one HMC sample (shape (26,)) into a params dict used by the 6DOF forward sim.

    Notes on naming:
      - HMC uses x_g, y_g, z_g (CG). We'll store as x_cg, y_cg, z_cg.
      - HMC provides z_b (CB z). We'll store as z_cb.
      - If your forward model expects additional fields, add them here once.
    """
    # fixed mass for now (you can later sample mass too if it becomes a parameter)
    m = float(m_fixed)

    p = {
        # Physical
        "m": m,
        "W": m * 9.81,                          # [N]
        "B": float(theta_row[idx["B"]]),        # [N]
        "I_xx": float(theta_row[idx["I_xx"]]),
        "I_yy": float(theta_row[idx["I_yy"]]),
        "I_zz": float(theta_row[idx["I_zz"]]),

        # CG / CB
        "x_cg": float(theta_row[idx["x_g"]]),
        "y_cg": float(theta_row[idx["y_g"]]),
        "z_cg": float(theta_row[idx["z_g"]]),
        "x_cb": 0.0,
        "y_cb": 0.0,
        "z_cb": float(theta_row[idx["z_b"]]),

        # Added mass derivatives
        "X_dot_u": float(theta_row[idx["X_dot_u"]]),
        "Y_dot_v": float(theta_row[idx["Y_dot_v"]]),
        "Z_dot_w": float(theta_row[idx["Z_dot_w"]]),
        "K_dot_p": float(theta_row[idx["K_dot_p"]]),
        "M_dot_q": float(theta_row[idx["M_dot_q"]]),
        "N_dot_r": float(theta_row[idx["N_dot_r"]]),

        # Linear damping
        "X_u": float(theta_row[idx["X_u"]]),
        "Y_v": float(theta_row[idx["Y_v"]]),
        "Z_w": float(theta_row[idx["Z_w"]]),
        "K_p": float(theta_row[idx["K_p"]]),
        "M_q": float(theta_row[idx["M_q"]]),
        "N_r": float(theta_row[idx["N_r"]]),

        # Quadratic damping
        "X_uu": float(theta_row[idx["X_uu"]]),
        "Y_vv": float(theta_row[idx["Y_vv"]]),
        "Z_ww": float(theta_row[idx["Z_ww"]]),
        "K_pp": float(theta_row[idx["K_pp"]]),
        "M_qq": float(theta_row[idx["M_qq"]]),
        "N_rr": float(theta_row[idx["N_rr"]]),
    }
    return p



# ============================================================
# === 3) Helpers (Full 6DOF Fossen) ==========================
# ============================================================

def skew(a: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix S(a) such that S(a)b = a × b."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    return np.array([[0.0, -az,  ay],
                     [az,  0.0, -ax],
                     [-ay, ax,  0.0]], dtype=float)

def R_b_to_n(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Body-to-NED rotation matrix for 3-2-1 (roll-pitch-yaw): R = Rz(psi) Ry(theta) Rx(phi)
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [ cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [ spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [   -sth,              cth*sphi,                 cth*cphi]
    ], dtype=float)

def T_omega(phi: float, theta: float) -> np.ndarray:
    """
    Euler angle rate mapping for 3-2-1:
        [phi_dot, theta_dot, psi_dot]^T = T(phi,theta) * [p,q,r]^T
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)

    # Guard against theta near ±90 deg
    if abs(cth) < 1e-6:
        cth = np.sign(cth) * 1e-6

    return np.array([
        [1.0, sphi*sth/cth,  cphi*sth/cth],
        [0.0,        cphi,         -sphi],
        [0.0, sphi/cth,      cphi/cth]
    ], dtype=float)

def J_eta(eta: np.ndarray) -> np.ndarray:
    """Kinematic transform: eta_dot = J(eta) * nu."""
    phi, theta, psi = float(eta[3]), float(eta[4]), float(eta[5])
    R = R_b_to_n(phi, theta, psi)
    T = T_omega(phi, theta)
    J = np.zeros((6, 6), dtype=float)
    J[0:3, 0:3] = R
    J[3:6, 3:6] = T
    return J

# --- Rigid-body mass matrix ---
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

# --- Added-mass matrix (diagonal) ---
def M_A(params: dict) -> np.ndarray:
    # Fossen convention: M_A = -diag(X_dot_u, Y_dot_v, ...)
    Xdu = float(params["X_dot_u"])
    Ydv = float(params["Y_dot_v"])
    Zdw = float(params["Z_dot_w"])
    Kdp = float(params["K_dot_p"])
    Mdq = float(params["M_dot_q"])
    Ndr = float(params["N_dot_r"])
    return -np.diag([Xdu, Ydv, Zdw, Kdp, Mdq, Ndr]).astype(float)

def M_total(params: dict) -> np.ndarray:
    return M_RB(params) + M_A(params)

# --- Rigid-body Coriolis matrix ---
def C_RB(nu: np.ndarray, params: dict) -> np.ndarray:
    """
    Rigid-body Coriolis/centripetal matrix C_RB(nu), matching Torch implementation exactly.

    Conventions:
      - nu = [u, v, w, p, q, r]
      - NED with z positive down
      - CG offsets: x_g, y_g, z_g in meters (body frame)
      - Inertia: I_xx, I_yy, I_zz about body axes

    Returns:
      C (6x6) such that coriolis term is C(nu) @ nu
    """
    u, v, w, p, q, r = [float(nu[i]) for i in range(6)]

    m = float(params["m"])
    Ixx = float(params["I_xx"])
    Iyy = float(params["I_yy"])
    Izz = float(params["I_zz"])

    xg = float(params["x_cg"])
    yg = float(params["y_cg"])
    zg = float(params["z_cg"])

    C = np.zeros((6, 6), dtype=float)

    # --- Top-left / translational coupling ---
    C[0, 1] = -m * r
    C[0, 2] =  m * q
    C[1, 0] =  m * r
    C[1, 2] = -m * p
    C[2, 0] = -m * q
    C[2, 1] =  m * p

    # --- Top-right (translation <- rotation) ---
    C[0, 3] =  m * (q * yg + r * zg)
    C[0, 4] = -m * (q * xg)
    C[0, 5] = -m * (r * xg)

    C[1, 3] = -m * (p * yg)
    C[1, 4] =  m * (p * xg + r * zg)
    C[1, 5] = -m * (r * yg)

    C[2, 3] = -m * (p * zg)
    C[2, 4] = -m * (q * zg)
    C[2, 5] =  m * (p * xg + q * yg)

    # --- Bottom-left (rotation <- translation) ---
    C[3, 0] = -m * (q * yg + r * zg)
    C[3, 1] =  m * (p * yg)
    C[3, 2] =  m * (p * zg)

    C[4, 0] =  m * (q * xg)
    C[4, 1] = -m * (p * xg + r * zg)
    C[4, 2] =  m * (q * zg)

    C[5, 0] =  m * (r * xg)
    C[5, 1] =  m * (r * yg)
    C[5, 2] = -m * (p * xg + q * yg)

    # --- Bottom-right (rotational) ---
    C[3, 4] =  Izz * r
    C[3, 5] = -Iyy * q

    C[4, 3] = -Izz * r
    C[4, 5] =  Ixx * p

    C[5, 3] =  Iyy * q
    C[5, 4] = -Ixx * p

    return C


# --- Added-mass Coriolis matrix (diagonal added mass) ---
def C_A(nu: np.ndarray, params: dict) -> np.ndarray:
    """
    Added-mass Coriolis matrix C_A(nu), matching your Torch implementation exactly.

    Conventions:
      - nu = [u, v, w, p, q, r]
      - params contain: X_dot_u, Y_dot_v, Z_dot_w, K_dot_p, M_dot_q, N_dot_r

    Returns:
      C_A (6x6) such that added-mass coriolis term is C_A(nu) @ nu
    """
    u, v, w, p, q, r = [float(nu[i]) for i in range(6)]

    Xdu = float(params["X_dot_u"])
    Ydv = float(params["Y_dot_v"])
    Zdw = float(params["Z_dot_w"])
    Kdp = float(params["K_dot_p"])
    Mdq = float(params["M_dot_q"])
    Ndr = float(params["N_dot_r"])

    # a-terms (translation-related)
    a1 = Xdu * u
    a2 = Ydv * v
    a3 = Zdw * w

    # b-terms (rotation-related)
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

# --- Damping (diagonal linear + quadratic) ---
def D_nu(nu: np.ndarray, params: dict) -> np.ndarray:
    """
    Matches Torch:
      D(ν) = -diag(lin + quad*|nu|)
    where lin = [X_u, Y_v, Z_w, K_p, M_q, N_r]
          quad = [X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr]
    """
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

    # Torch: D = -diag_embed(diag_entries)
    return -np.diag(diag_entries)

# --- Restoring forces (weight/buoyancy) ---
def g_eta(eta: np.ndarray, params: dict) -> np.ndarray:
    """
    Restoring vector g(eta), NED (z down), using official parameter keys:
      CG: x_cg,y_cg,z_cg
      CB: x_cb,y_cb,z_cb
    """
    phi   = float(eta[3])
    theta = float(eta[4])

    g0 = 9.8  # match Torch constant if desired

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

    cth, sth  = np.cos(theta), np.sin(theta)
    cphi, sphi = np.cos(phi), np.sin(phi)

    return np.array([
        WB * sth,
        -WB * cth * sphi,
        -WB * cth * cphi,
        -yW_yB * cth * cphi + zW_zB * cth * sphi,
         zW_zB * sth + xW_xB * cth * cphi,
        -xW_xB * cth * sphi - yW_yB * sth
    ], dtype=float)


def predict_nu_dot_from_csv_states(
    nu_meas: np.ndarray,     # (N,6) measured ν from CSV
    eta_meas: np.ndarray,    # (N,6) measured η from CSV
    tau: np.ndarray,         # (N,6) applied τ from CSV
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
) -> np.ndarray:
    """
    No integration. Algebraic prediction at each timestep:
        nu_dot_hat = M^{-1} [ tau - C(nu)nu - D(nu)nu - g(eta) ]

    Uses measured nu_meas[k], eta_meas[k], tau[k].
    Returns nu_dot_hat with shape (N,6).
    """
    N = nu_meas.shape[0]
    nu_dot_hat = np.zeros((N, 6), dtype=float)

    M = M_total(params)  # constant if params constant

    for k in range(N):
        nu_k = nu_meas[k, :]
        eta_k = eta_meas[k, :]

        C = C_total(nu_k, params, include_ca=include_ca)
        D = D_nu(nu_k, params)
        gvec = g_eta(eta_k, params) if include_g else np.zeros(6, dtype=float)

        rhs = tau[k, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot_hat[k, :] = np.linalg.solve(M, rhs)

    return nu_dot_hat



# --- Full 6DOF forward integrator (Euler) ---
def forward_integrate_6dof(
    t: np.ndarray,
    nu0: np.ndarray,
    eta0: np.ndarray,
    tau: np.ndarray,          # (N,6)
    params: dict,
    include_ca: bool = False,
    include_g: bool = False,
):
    """
    Forward integrate full 6DOF (Euler / semi-implicit for eta):
        M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau
        eta_dot = J(eta) nu
    """
    N = len(t)
    nu_pred = np.zeros((N, 6), dtype=float)
    eta_pred = np.zeros((N, 6), dtype=float)
    nu_dot_pred = np.zeros((N, 6), dtype=float)

    nu_pred[0, :] = np.asarray(nu0, dtype=float).reshape(6,)
    eta_pred[0, :] = np.asarray(eta0, dtype=float).reshape(6,)

    M = M_total(params)

    for k in range(1, N):
        dt = DT_FIXED if FORCE_FIXED_DT else float(t[k] - t[k - 1])

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

        rhs = tau[k - 1, :] - (C @ nu_k) - (D @ nu_k) - gvec
        nu_dot = np.linalg.solve(M, rhs)

        # integrate nu (explicit Euler)
        nu_kp1 = nu_k + nu_dot * dt
        nu_pred[k, :] = nu_kp1
        nu_dot_pred[k, :] = nu_dot

        # integrate eta (use updated nu for better stability)
        J = J_eta(eta_k)
        eta_dot = J @ nu_kp1
        eta_pred[k, :] = eta_k + eta_dot * dt

    return nu_pred, eta_pred, nu_dot_pred



def residual_stats(residual: np.ndarray):
    """Return mean and RMSE, ignoring non-finite values."""
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(r))
    rmse = float(np.sqrt(np.mean(r**2)))
    return mean, rmse

def posterior_predictive_band_u_6dof(
    t: np.ndarray,
    tau: np.ndarray,          # (T,6)
    nu0: np.ndarray,          # (6,)
    eta0: np.ndarray,         # (6,)
    theta_plot: np.ndarray,   # (Nsamp, n_params)
    param_names: list,        # length n_params
    params_base: dict,        # contains fixed keys (m, x_cg, ..., x_cb, y_cb, z_cb, etc.)
    include_ca: bool = False,
    include_g: bool = False,
):
    """
    For many posterior samples, forward integrate full 6DOF and return band for u(t).
    """
    Ns = theta_plot.shape[0]
    T = len(t)
    U = np.zeros((Ns, T), dtype=float)

    # map name->col index once
    idx = {name: i for i, name in enumerate(param_names)}

    for j in range(Ns):
        # start from base, then overwrite sampled params that exist in idx
        p = dict(params_base)

        # overwrite any sampled parameters present
        for name, col in idx.items():
            p[name] = float(theta_plot[j, col])

        nu_pred, eta_pred, nu_dot_pred = forward_integrate_6dof(
            t=t,
            nu0=nu0,
            eta0=eta0,
            tau=tau,
            params=p,
            include_ca=include_ca,
            include_g=include_g,
        )

        U[j, :] = nu_pred[:, 0]  # surge u

    p05 = np.percentile(U, 5, axis=0)
    p50 = np.percentile(U, 50, axis=0)
    p95 = np.percentile(U, 95, axis=0)
    return p05, p50, p95

# ============================================================
# === 4) Run predictions (Full 6DOF, selectable DOF) =========
# ============================================================
pred = {}
residuals = {}
stats = {}

for label, p in PARAM_SETS.items():

    nu_pred, eta_pred, nu_dot_pred = forward_integrate_6dof(
        t=t,
        nu0=nu_meas[0, :],
        eta0=eta_meas[0, :],
        tau=tau,
        params=p,
        include_ca=INCLUDE_CA,
        include_g=INCLUDE_G,
    )

    if ACTIVE_DOF == "ALL":
        pred[label] = nu_pred                      # shape (N,6)
        residuals[label] = nu_meas - nu_pred       # shape (N,6)

        # optional: per-DOF stats
        stats[label] = {
            dof: residual_stats(residuals[label][:, i])
            for dof, i in DOF_INDEX.items()
        }

    else:
        i = DOF_INDEX[ACTIVE_DOF]

        pred[label] = nu_pred[:, i]
        residuals[label] = nu_meas[:, i] - pred[label]
        stats[label] = residual_stats(residuals[label])


# ============================================================
# === 4b) Posterior predictive band from HMC samples =========
# ============================================================
i_dof = DOF_INDEX[ACTIVE_DOF]

Ns = theta_plot.shape[0]
T  = len(t)

U = np.zeros((Ns, T), dtype=float)

# map parameter names -> column index once
param_idx = {name: i for i, name in enumerate(param_names)}

for j in range(Ns):
    # start from MAP (or base) params
    p = dict(params_map)

    # overwrite with sampled parameters
    for name, idx in param_idx.items():
        p[name] = float(theta_plot[j, idx])

    nu_pred, eta_pred, nu_dot_pred = forward_integrate_6dof(
        t=t,
        nu0=nu_meas[0, :],
        eta0=eta_meas[0, :],
        tau=tau,
        params=p,
        include_ca=INCLUDE_CA,
        include_g=INCLUDE_G,
    )

    U[j, :] = nu_pred[:, i_dof]

u_p05 = np.percentile(U, 5, axis=0)
u_p50 = np.percentile(U, 50, axis=0)
u_p95 = np.percentile(U, 95, axis=0)

band_width = u_p95 - u_p05

print("band width: min/mean/max =",
      np.nanmin(band_width),
      np.nanmean(band_width),
      np.nanmax(band_width))

print("finite fractions:",
      "p05=", np.isfinite(u_p05).mean(),
      "p50=", np.isfinite(u_p50).mean(),
      "p95=", np.isfinite(u_p95).mean())



# ============================================================
# === 4c) No-integration accel check (CSV ν,η,τ -> ν̇_hat) ====
# ============================================================
accel_pred = {}
accel_residuals = {}
accel_stats = {}

for label, p in PARAM_SETS.items():
    nu_dot_hat = predict_nu_dot_from_csv_states(
        nu_meas=nu_meas,
        eta_meas=eta_meas,
        tau=tau,
        params=p,
        include_ca=INCLUDE_CA,
        include_g=INCLUDE_G,
    )

    if ACTIVE_DOF == "ALL":
        accel_pred[label] = nu_dot_hat
        accel_residuals[label] = nu_dot_meas - nu_dot_hat
        accel_stats[label] = {
            dof: residual_stats(accel_residuals[label][:, i])
            for dof, i in DOF_INDEX.items()
        }
    else:
        i = DOF_INDEX[ACTIVE_DOF]
        accel_pred[label] = nu_dot_hat[:, i]
        accel_residuals[label] = nu_dot_meas[:, i] - accel_pred[label]
        accel_stats[label] = residual_stats(accel_residuals[label])


# ============================================================
# === 5) Plot (DOF-aware) ====================================
# ============================================================

DOF_META = {
    "X": {"name": "Surge", "sym": "u", "tau": "X", "unit": "m/s",  "tau_unit": "N",   "ylim": (-2, 2)},
    "Y": {"name": "Sway",  "sym": "v", "tau": "Y", "unit": "m/s",  "tau_unit": "N",   "ylim": (-2, 2)},
    "Z": {"name": "Heave", "sym": "w", "tau": "Z", "unit": "m/s",  "tau_unit": "N",   "ylim": (-2, 2)},
    "K": {"name": "Roll",  "sym": "p", "tau": "K", "unit": "rad/s","tau_unit": "N·m", "ylim": (-3, 3)},
    "M": {"name": "Pitch", "sym": "q", "tau": "M", "unit": "rad/s","tau_unit": "N·m", "ylim": (-3, 3)},
    "N": {"name": "Yaw",   "sym": "r", "tau": "N", "unit": "rad/s","tau_unit": "N·m", "ylim": (-3, 3)},
}

if ACTIVE_DOF == "ALL":
    raise ValueError("This plotting block expects a single ACTIVE_DOF (X/Y/Z/K/M/N).")

i_dof = DOF_INDEX[ACTIVE_DOF]
meta = DOF_META[ACTIVE_DOF]

# Select measured signals for the active DOF
nu_meas_dof = nu_meas[:, i_dof]
tau_dof     = tau[:, i_dof]

# Build a consistent title line about toggles
toggle_line = f"Includes C_RB{' + C_A' if INCLUDE_CA else ''}{', includes g(eta)' if INCLUDE_G else ', excludes g(eta)'}"

plt.figure(figsize=(14, 10))

# 1) Applied tau component
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau_dof, "r-", label=f"{meta['tau']} applied τ")
ax1.set_ylabel(f"{meta['tau']} [{meta['tau_unit']}]")
ax1.set_title(f"{meta['name']} ({ACTIVE_DOF}) Post-Check: measured {meta['sym']} vs predicted {meta['sym']} "
              f"(MLE / MAP / HMC MAP)\n{toggle_line}")
ax1.grid(True)
ax1.legend()

# 2) measured vs predicted + band
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t_rel, nu_meas_dof, linewidth=2, label=f"{meta['sym']} measured")

# predictions from Section 4 (pred dict or u_pred dict)
ax2.plot(t_rel, pred["MLE"], linewidth=2, label=f"{meta['sym']} predicted (MLE)")
ax2.plot(t_rel, pred["MAP"], linewidth=2, label=f"{meta['sym']} predicted (MAP)")
ax2.plot(t_rel, pred["HMC_MAP"], linewidth=2, label=f"{meta['sym']} predicted (HMC MAP)")

# HMC posterior predictive band (from Section 4b)
ax2.fill_between(t_rel, u_p05, u_p95, alpha=0.2, label="HMC 90% predictive band")
ax2.plot(t_rel, u_p50, linewidth=2, label="HMC median prediction")

ax2.set_ylabel(f"{meta['sym']} [{meta['unit']}]")
ax2.set_ylim(list(meta["ylim"]))
ax2.grid(True)
ax2.legend()

# 3) residuals
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
for label in ["MLE", "MAP", "HMC_MAP"]:
    mu, rmse = stats[label]
    ax3.plot(t_rel, residuals[label],
             label=f"Residual ({label})  mean={mu:+.4f}, rmse={rmse:.4f}")
    ax3.axhline(mu, linestyle="--", linewidth=1)

ax3.set_ylabel(f"Residual [{meta['unit']}]")
ax3.set_xlabel("Time [s]")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

# ============================================================
# === 5c) Plot accel consistency check (no integration) =======
# ============================================================
ACCEL_META = {
    "X": {"name": "Surge", "sym": "u̇", "unit": "m/s²",   "ylim": None},
    "Y": {"name": "Sway",  "sym": "v̇", "unit": "m/s²",   "ylim": None},
    "Z": {"name": "Heave", "sym": "ẇ", "unit": "m/s²",   "ylim": None},
    "K": {"name": "Roll",  "sym": "ṗ", "unit": "rad/s²", "ylim": None},
    "M": {"name": "Pitch", "sym": "q̇", "unit": "rad/s²", "ylim": None},
    "N": {"name": "Yaw",   "sym": "ṙ", "unit": "rad/s²", "ylim": None},
}

if ACTIVE_DOF == "ALL":
    raise ValueError("Accel plotting block expects a single ACTIVE_DOF (X/Y/Z/K/M/N).")

i_dof = DOF_INDEX[ACTIVE_DOF]
ameta = ACCEL_META[ACTIVE_DOF]

nu_dot_meas_dof = nu_dot_meas[:, i_dof]

plt.figure(figsize=(14, 8))

# 1) accel overlay
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t_rel, nu_dot_meas_dof, linewidth=2, label=f"{ameta['sym']} truth (CSV)")

ax1.plot(t_rel, accel_pred["MLE"], linewidth=2, label=f"{ameta['sym']} predicted (MLE)")
ax1.plot(t_rel, accel_pred["MAP"], linewidth=2, label=f"{ameta['sym']} predicted (MAP)")
ax1.plot(t_rel, accel_pred["HMC_MAP"], linewidth=2, label=f"{ameta['sym']} predicted (HMC MAP)")

ax1.set_ylabel(f"{ameta['sym']} [{ameta['unit']}]")
ax1.set_title(f"{ameta['name']} accel consistency check (no integration): "
              f"CSV ν,η,τ → predicted ν̇\n{toggle_line}")
ax1.grid(True)
ax1.legend()

# 2) accel residuals
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for label in ["MLE", "MAP", "HMC_MAP"]:
    mu, rmse = accel_stats[label]
    ax2.plot(t_rel, accel_residuals[label],
             label=f"Residual ({label})  mean={mu:+.4e}, rmse={rmse:.4e}")
    ax2.axhline(mu, linestyle="--", linewidth=1)

ax2.set_ylabel(f"Residual [{ameta['unit']}]")
ax2.set_xlabel("Time [s]")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


dt_vec = np.diff(t)
print("dt stats:",
      np.min(dt_vec),
      np.mean(dt_vec),
      np.max(dt_vec))
