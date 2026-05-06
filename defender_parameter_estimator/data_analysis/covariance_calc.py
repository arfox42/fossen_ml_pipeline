#!/usr/bin/env python3
"""
estimate_Q_nudot_from_single_dof_runs.py
---------------------------------------
Compute a diagonal acceleration process covariance Q_{nu_dot} for an EKF
using single-DOF excitation datasets.

For each DOF dataset:
  1) Load CSV (same column convention as post_check_full6dof_single_dof.py)
  2) Compute tau_hat via inverse dynamics:
        tau_hat = M nu_dot + C(nu)nu + D(nu)nu + g(eta)
  3) Residual: r_tau = tau_meas - tau_hat
  4) Estimate sigma^2_tau for the active DOF (N-p correction)
  5) Convert to sigma^2_nu_dot via:
        sigma^2_nu_dot_i = sigma^2_tau_i / (M_ii)^2
  6) Assemble Q_nudot_diag = diag([udot, vdot, wdot, pdot, qdot, rdot])

NOTES:
- This assumes your disturbance enters like an additive wrench disturbance.
- Only the ACTIVE DOF in each dataset is considered reliable (strongly excited).
"""

import numpy as np

# ============================================================
# === User inputs ============================================
# ============================================================n

# Provide 1 dataset per DOF (single-DOF excitation runs)
DOF_FILES = {
    "X": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/X_Data/Tank_data/defender_data_x_run_1_ahrs_corrected.csv",
    "Y": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Y_Data/Tank_data/defender_data_y_run_1_ahrs_corrected.csv",
    "Z": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Z_Data/Tank_data/defender_data_z_run_savgol.csv",
    "K": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/K_Data/Tank_data/defender_data_k_run_savgol.csv",
    "M": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/M_Data/Tank_data/defender_data_m_run_savgol.csv",
    "N": "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/N_Data/Tank_data/defender_data_n_run_1_only_mocap_data_savgol.csv",
}

# Physics toggles (match what you use in post-check)
INCLUDE_CA = True
INCLUDE_G  = True

# Optional: only use samples with |tau_active| above threshold (helps avoid near-zero segments)
USE_TAU_THRESHOLD = False
TAU_ABS_MIN = 5.0

# ============================================================
# === DOF index maps =========================================
# ============================================================

DOF_INDEX = {"X": 0, "Y": 1, "Z": 2, "K": 3, "M": 4, "N": 5}
DOF_LABEL = {0: "tau_x", 1: "tau_y", 2: "tau_z", 3: "tau_k", 4: "tau_m", 5: "tau_n"}

# ============================================================
# === Parameters (paste your MLE library here) ================
# ============================================================

G = 9.8
params_base = {"W": 23.89 * G}

# Replace this params_mle block with the numbers you want to use.
# You can paste directly from your post-check script's params_mle.
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

# ============================================================
# === Fossen helpers (copied from your post-check) ============
# ============================================================

def skew(a: np.ndarray) -> np.ndarray:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    return np.array([[0.0, -az,  ay],
                     [az,  0.0, -ax],
                     [-ay, ax,  0.0]], dtype=float)

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

def tau_hat_from_states(nu: np.ndarray, nu_dot: np.ndarray, eta: np.ndarray, params: dict,
                        include_ca: bool, include_g: bool) -> np.ndarray:
    M = M_total(params)
    C = C_total(nu, params, include_ca=include_ca)
    D = D_nu(nu, params)
    gvec = g_eta(eta, params) if include_g else np.zeros(6, dtype=float)
    return (M @ nu_dot) + (C @ nu) + (D @ nu) + gvec

# ============================================================
# === CSV loader (same convention as your post-check) =========
# ============================================================

def load_csv(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)
    if data.ndim != 2 or data.shape[1] < 25:
        raise RuntimeError(f"CSV malformed or too few columns: {csv_path}, shape={data.shape}")

    t = data[:, 0].astype(float)

    nu_dot = np.vstack([data[:, i].astype(float) for i in range(1, 7)]).T
    nu     = np.vstack([data[:, i].astype(float) for i in range(7, 13)]).T
    eta    = np.vstack([data[:, i].astype(float) for i in range(13, 19)]).T
    tau    = np.vstack([data[:, i].astype(float) for i in range(19, 25)]).T

    # validity columns (25..30) if present
    pose_valid = twist_valid = wrench_valid = None
    if data.shape[1] >= 31:
        pose_valid   = data[:, 25].astype(float)
        twist_valid  = data[:, 27].astype(float)
        wrench_valid = data[:, 29].astype(float)

    return t, nu_dot, nu, eta, tau, pose_valid, twist_valid, wrench_valid

def build_valid_mask(nu, nu_dot, eta, tau, pose_valid, twist_valid, wrench_valid):
    m = (
        np.isfinite(nu).all(axis=1) &
        np.isfinite(nu_dot).all(axis=1) &
        np.isfinite(eta).all(axis=1) &
        np.isfinite(tau).all(axis=1)
    )
    # If validity flags are present, require them (like your script)
    if pose_valid is not None:
        m &= (pose_valid > 0.5)
    if twist_valid is not None:
        m &= (twist_valid > 0.5)
    if wrench_valid is not None:
        m &= (wrench_valid > 0.5)
    return m

# ============================================================
# === Main: per DOF residual variance -> Q_nudot diag =========
# ============================================================

def estimate_sigma2_tau_active_mle(res_tau_active: np.ndarray):
    """
    MLE variance estimate for nonlinear model residuals:
        sigma^2 = (1/N) * sum (r - mean(r))^2
    Returns (sigma2, N, mean_residual)
    """
    r = res_tau_active[np.isfinite(res_tau_active)]
    N = r.size
    if N < 20:
        return np.nan, N, np.nan

    mu = float(np.mean(r))
    sigma2 = float(np.mean((r - mu)**2))   # <-- 1/N
    return sigma2, N, mu

def main():
    # Inertia used for mapping tau variance -> nu_dot variance
    M = M_total(params_mle)
    Mdiag = np.diag(M).astype(float)  # [M11, M22, M33, M44, M55, M66]

    print("\n=== Using diagonal inertia terms from params_mle ===")
    for k, val in zip(["M11","M22","M33","M44","M55","M66"], Mdiag):
        print(f"{k}: {val:.6f}")

    sigma2_tau = np.full(6, np.nan, dtype=float)
    sigma2_nudot = np.full(6, np.nan, dtype=float)
    N_used = np.zeros(6, dtype=int)

    for dof, path in DOF_FILES.items():
        i = DOF_INDEX[dof]
        print(f"\n--- DOF {dof} ({DOF_LABEL[i]}) ---")
        print(f"file: {path}")

        t, nu_dot, nu, eta, tau, pose_valid, twist_valid, wrench_valid = load_csv(path)
        mask = build_valid_mask(nu, nu_dot, eta, tau, pose_valid, twist_valid, wrench_valid)

        if USE_TAU_THRESHOLD:
            mask &= (np.abs(tau[:, i]) >= TAU_ABS_MIN)

        idx = np.where(mask)[0]
        if idx.size < 50:
            print(f"[WARN] Too few valid samples after masking: {idx.size}")
            continue

        # Compute tau_hat sample-by-sample (inverse dynamics)
        tau_hat = np.full_like(tau, np.nan, dtype=float)
        for k in idx:
            tau_hat[k, :] = tau_hat_from_states(
                nu=nu[k, :],
                nu_dot=nu_dot[k, :],
                eta=eta[k, :],
                params=params_mle,
                include_ca=INCLUDE_CA,
                include_g=INCLUDE_G,
            )

        res_tau = tau - tau_hat  # [N,6]
        res_active = res_tau[:, i]

        s2_tau, Nact, mu_tau = estimate_sigma2_tau_active_mle(res_active)

        sigma2_tau[i] = s2_tau
        N_used[i] = Nact

        if np.isfinite(s2_tau):
            Mi = float(Mdiag[i])
            sigma2_nudot[i] = s2_tau / (Mi ** 2)
            print(f"mean(res_tau[{DOF_LABEL[i]}])   = {mu_tau:+.6f}")
            print(f"sigma^2_tau[{DOF_LABEL[i]}]    = {s2_tau:.6f}  (1/N)")
            print(f"sigma^2_nu_dot[{i}]            = {sigma2_nudot[i]:.6e}")
        else:
            print("[WARN] sigma^2_tau could not be computed (insufficient data).")

    Q_nudot = np.diag(sigma2_nudot)

    print("\n==================== RESULT ====================")
    print("N_used per channel:", N_used.tolist())
    print("sigma2_tau   :", sigma2_tau)
    print("sigma2_nudot :", sigma2_nudot)
    print("\nQ_nudot (diagonal):")
    print(Q_nudot)
    print("================================================\n")

    # Save results
    np.save("Q_nudot_diag.npy", Q_nudot)
    np.save("sigma2_tau.npy", sigma2_tau)
    np.save("sigma2_nudot.npy", sigma2_nudot)
    print("[saved] Q_nudot_diag.npy, sigma2_tau.npy, sigma2_nudot.npy")

if __name__ == "__main__":
    main()