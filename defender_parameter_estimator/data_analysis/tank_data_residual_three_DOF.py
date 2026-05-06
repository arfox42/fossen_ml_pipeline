#!/usr/bin/env python3
"""
post_check_full6dof_multi_dof.py
--------------------------------
Forward-integrates full 6DOF, then plots MULTIPLE DOFs (e.g., X, Z, N)
as separate subplots: truth vs prediction overlays.

Overlays:
  - MLE
  - MAP
  - HMC median + 90% predictive band

Notes:
  - HMC band is computed from HMC samples by overwriting params_map keys.
  - Divergent sample trajectories are rejected.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# === 0) Config / Flags ======================================
# ============================================================

ACTIVE_DOFS = ["X", "Z", "N"]  # plot these DOFs (3 rows)

csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Coupled Maneuvers/defender_data_teleop_circle_z_sin.csv"

# Physics toggles
INCLUDE_CA = True
INCLUDE_G  = True

# Integrator timing
FORCE_FIXED_DT = True
DT_FIXED = 0.01

# Which overlays to show
PLOT_MLE = True
PLOT_MAP = True
PLOT_HMC = True

# HMC predictive band
HMC_SAMPLES_PATHS = {
    "X": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_surge_samples.pt",
    "Z": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_heave_samples.pt",
    "N": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_yaw_samples.pt",
}

HMC_BAND_NPLOT = 100

# Divergence filtering thresholds (tune)
NU_ABS_MAX = {
    "X": 10.0,  # m/s
    "Y": 10.0,
    "Z": 10.0,
    "K": 30.0,  # rad/s (probably never this high)
    "M": 30.0,
    "N": 30.0,
}
NAN_FRAC_MAX = 0.01

# ============================================================
# === Index maps =============================================
# ============================================================
DOF_INDEX = {"X":0,"Y":1,"Z":2,"K":3,"M":4,"N":5}

DOF_META = {
    "X": {"name": "Surge", "sym": "u", "tau": "X", "unit": "m/s",   "tau_unit": "N"},
    "Y": {"name": "Sway",  "sym": "v", "tau": "Y", "unit": "m/s",   "tau_unit": "N"},
    "Z": {"name": "Heave", "sym": "w", "tau": "Z", "unit": "m/s",   "tau_unit": "N"},
    "K": {"name": "Roll",  "sym": "p", "tau": "K", "unit": "rad/s", "tau_unit": "N·m"},
    "M": {"name": "Pitch", "sym": "q", "tau": "M", "unit": "rad/s", "tau_unit": "N·m"},
    "N": {"name": "Yaw",   "sym": "r", "tau": "N", "unit": "rad/s", "tau_unit": "N·m"},
}

# ============================================================
# === 1) Load CSV ============================================
# ============================================================
data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)

t = data[:, 0].astype(float)
t_rel = t - t[0]

# Measured body velocities ν = [u v w p q r]
nu_meas = np.vstack((
    data[:, 7].astype(float),
    data[:, 8].astype(float),
    data[:, 9].astype(float),
    data[:,10].astype(float),
    data[:,11].astype(float),
    data[:,12].astype(float),
)).T

# Applied forces/moments τ = [X Y Z K M N]
tau = np.vstack((
    data[:,19].astype(float),
    data[:,20].astype(float),
    data[:,21].astype(float),
    data[:,22].astype(float),
    data[:,23].astype(float),
    data[:,24].astype(float),
)).T

# Position / attitude η = [x y z phi theta psi]
eta_meas = np.vstack((
    data[:,13].astype(float),
    data[:,14].astype(float),
    data[:,15].astype(float),
    data[:,16].astype(float),
    data[:,17].astype(float),
    data[:,18].astype(float),
)).T

assert np.all(np.isfinite(nu_meas[0, :])), "nu_meas[0] contains NaN/Inf"
assert np.all(np.isfinite(eta_meas[0, :])), "eta_meas[0] contains NaN/Inf"
assert np.all(np.isfinite(tau[0, :])), "tau[0] contains NaN/Inf"

# ============================================================
# === 2) Parameters ==========================================
# ============================================================
G = 9.8

params_base = {"W": 23.89 * G}

# --- MLE (your hybrid example) ---
params_mle = {
    **params_base,
    "m": 23.89, "B": 236.00, "I_xx": 0.5, "I_yy": 1.76, "I_zz": 2.13,
    "x_cg": 0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,

    "X_dot_u": -33.61, "Y_dot_v": -31.56, "Z_dot_w": -79.58,
    "K_dot_p": -0.1, "M_dot_q": -0.46, "N_dot_r": -0.70,

    "X_u": -16.49, "X_uu": -42.49,
    "Y_v": -34.05, "Y_vv": -108.74,
    "Z_w": -35.66, "Z_ww": -128.31,
    "K_p": -1.24, "K_pp": -0.08,
    "M_q": -2.08, "M_qq": -1.61,
    "N_r": -2.88, "N_rr": -2.69,
}

# --- MAP (your example) ---
params_map = {
    **params_base,
    "m": 23.89, "B": 235.97, "I_xx": 0.41, "I_yy": 1.31, "I_zz": 1.46,
    "x_cg": -0.0, "y_cg": 0.0, "z_cg": 0.0,
    "x_cb": 0.0, "y_cb": 0.0, "z_cb": -0.03,

    "X_dot_u": -32.41, "Y_dot_v": -16.78, "Z_dot_w": -77.77,
    "K_dot_p": -0.22, "M_dot_q": -0.91, "N_dot_r": -1.32,

    "X_u": -1.01, "X_uu": -62.05,
    "Y_v": -0.93, "Y_vv": -137.19,
    "Z_w": -35.30, "Z_ww": -126.63,
    "K_p": -1.14, "K_pp": -0.2,
    "M_q": -1.05, "M_qq": -2.91,
    "N_r": -0.99, "N_rr": -3.51,
}

PARAM_SETS = {"MLE": params_mle, "MAP": params_map}

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

def forward_integrate_6dof(t, nu0, eta0, tau, params, include_ca=False, include_g=False):
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

        nu_kp1 = nu_k + nu_dot * dt
        nu_pred[k, :] = nu_kp1
        nu_dot_pred[k, :] = nu_dot

        J = J_eta(eta_k)
        eta_dot = J @ nu_kp1
        eta_pred[k, :] = eta_k + eta_dot * dt

    return nu_pred, eta_pred, nu_dot_pred

# ============================================================
# === 4) Run forward predictions (MLE / MAP) =================
# ============================================================
pred = {}  # pred[label] = full nu_pred (N,6)

if PLOT_MLE:
    nu_pred, _, _ = forward_integrate_6dof(
        t=t, nu0=nu_meas[0], eta0=eta_meas[0], tau=tau, params=params_mle,
        include_ca=INCLUDE_CA, include_g=INCLUDE_G
    )
    pred["MLE"] = nu_pred

if PLOT_MAP:
    nu_pred, _, _ = forward_integrate_6dof(
        t=t, nu0=nu_meas[0], eta0=eta_meas[0], tau=tau, params=params_map,
        include_ca=INCLUDE_CA, include_g=INCLUDE_G
    )
    pred["MAP"] = nu_pred

# ============================================================
# === 5) HMC band: compute p05/p50/p95 for requested DOFs =====
# ============================================================
def compute_hmc_band_for_dof(
    dof: str,
    pt_path: str,
    t: np.ndarray,
    nu0: np.ndarray,
    eta0: np.ndarray,
    tau: np.ndarray,
    params_baseline: dict,
    include_ca: bool,
    include_g: bool,
    nplot: int = 100,
    nan_frac_max: float = 0.01,
    abs_max: float = 10.0,
):
    """
    Returns (p05, p50, p95, kept, rejected) for a single DOF using that DOF's .pt file.
    """
    # --- load samples ---
    hmc_data = torch.load(pt_path, map_location="cpu")
    theta_samples = hmc_data["samples"].detach().cpu().numpy()
    param_names = list(hmc_data["param_names"])
    param_idx = {name: i for i, name in enumerate(param_names)}

    # --- subsample ---
    if theta_samples.shape[0] > nplot:
        sel = np.random.choice(theta_samples.shape[0], size=nplot, replace=False)
        theta_plot = theta_samples[sel]
    else:
        theta_plot = theta_samples

    i_dof = DOF_INDEX[dof]
    U_list = []
    kept = 0
    rejected = 0

    for j in range(theta_plot.shape[0]):
        p = dict(params_baseline)  # start from MAP baseline (or whatever baseline you want)

        # overwrite only keys that exist
        for name, col in param_idx.items():
            if name in p:
                p[name] = float(theta_plot[j, col])

        nu_pred, _, _ = forward_integrate_6dof(
            t=t,
            nu0=nu0,
            eta0=eta0,
            tau=tau,
            params=p,
            include_ca=include_ca,
            include_g=include_g,
        )

        u = nu_pred[:, i_dof].astype(float)
        nan_frac = np.mean(~np.isfinite(u))
        umax = np.nanmax(np.abs(u)) if np.any(np.isfinite(u)) else np.inf

        if (nan_frac > nan_frac_max) or (not np.isfinite(umax)) or (umax > abs_max):
            rejected += 1
            continue

        kept += 1
        U_list.append(u)

    if len(U_list) < 10:
        return None, None, None, kept, rejected

    U = np.vstack(U_list)  # (Nkeep, T)
    p05 = np.nanpercentile(U, 5, axis=0)
    p50 = np.nanpercentile(U, 50, axis=0)
    p95 = np.nanpercentile(U, 95, axis=0)
    return p05, p50, p95, kept, rejected


# ============================================================
# === 6) Plot: 3 rows (truth vs predictions + HMC band) =======
# ============================================================

toggle_line = f"Includes C_RB{' + C_A' if INCLUDE_CA else ''}{', includes g(eta)' if INCLUDE_G else ', excludes g(eta)'}"

# ------------------------------------------------------------
# Precompute HMC bands ONCE (per DOF), so we don't recompute
# them inside each subplot.
# band[d] = {"p05":..., "p50":..., "p95":...}
# ------------------------------------------------------------
band = {d: {"p05": None, "p50": None, "p95": None} for d in ACTIVE_DOFS}

if PLOT_HMC:
    for d in ACTIVE_DOFS:
        if d not in HMC_SAMPLES_PATHS:
            print(f"[HMC band] No .pt path provided for DOF {d}; skipping.")
            continue

        p05, p50, p95, kept, rejected = compute_hmc_band_for_dof(
            dof=d,
            pt_path=HMC_SAMPLES_PATHS[d],
            t=t,
            nu0=nu_meas[0],
            eta0=eta_meas[0],
            tau=tau,
            params_baseline=params_map,
            include_ca=INCLUDE_CA,
            include_g=INCLUDE_G,
            nplot=HMC_BAND_NPLOT,
            nan_frac_max=NAN_FRAC_MAX,
            abs_max=NU_ABS_MAX.get(d, 10.0),
        )

        band[d]["p05"] = p05
        band[d]["p50"] = p50
        band[d]["p95"] = p95

        print(f"[HMC band:{d}] kept {kept}/{kept + rejected} (rejected {rejected}) from {HMC_SAMPLES_PATHS[d]}")

# ------------------------------------------------------------
# Plot: one row per DOF
# ------------------------------------------------------------
fig, axes = plt.subplots(len(ACTIVE_DOFS), 1, figsize=(14, 4.2 * len(ACTIVE_DOFS)), sharex=True)
if len(ACTIVE_DOFS) == 1:
    axes = [axes]

for ax, d in zip(axes, ACTIVE_DOFS):
    meta = DOF_META[d]
    ii = DOF_INDEX[d]

    # truth
    ax.plot(t_rel, nu_meas[:, ii], linewidth=2.5, label=f"{meta['sym']} truth")

    # MLE / MAP overlays
    if "MLE" in pred:
        ax.plot(t_rel, pred["MLE"][:, ii], linewidth=2.0, label="MLE")
    if "MAP" in pred:
        ax.plot(t_rel, pred["MAP"][:, ii], linewidth=2.0, label="MAP")

    # HMC band + median (for THIS DOF only)
    if PLOT_HMC and (band[d]["p05"] is not None):
        ax.fill_between(
            t_rel,
            band[d]["p05"],
            band[d]["p95"],
            alpha=0.25,
            label="HMC 90% band",
            zorder=10,
        )
        ax.plot(
            t_rel,
            band[d]["p50"],
            linestyle="--",
            linewidth=2.5,
            label="HMC median",
            zorder=11,
        )

    ax.set_title(f"{meta['name']} ({d})")
    ax.set_ylabel(f"{meta['sym']} [{meta['unit']}]")
    ax.grid(True)
    ax.legend(loc="best")

axes[0].set_title(f"Multi-DOF Post-Check: truth vs forward prediction\n{toggle_line}", pad=18)
axes[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()


# ============================================================
# === 7) dt sanity print =====================================
# ============================================================
dt_vec = np.diff(t)
print("dt stats:", np.min(dt_vec), np.mean(dt_vec), np.max(dt_vec))
