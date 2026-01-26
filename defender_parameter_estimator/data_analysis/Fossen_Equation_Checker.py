#!/usr/bin/env python3
from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

@dataclass
class ROVParams:
    # Inertia
    I_xx: float = 1.0
    I_yy: float = 1.0
    I_zz: float = 1.0

    # CG / CB (body, NED)
    x_cg: float = 0.0
    y_cg: float = 0.0
    z_cg: float = 0.0
    x_cb: float = 0.0
    y_cb: float = 0.0
    z_cb: float = -0.05  # CB above CG in NED

    # Damping (expected NEGATIVE)
    X_u: float = -4.66
    X_uu: float = -51.5
    Y_v: float = -8.25
    Y_vv: float = -102.006
    Z_w: float = -14.17
    Z_ww: float = -155.8358
    K_p: float = -1.5
    K_pp: float = -2.1
    M_q: float = -2.9
    M_qq: float = -14.6
    N_r: float = -10.343
    N_rr: float = -8.8

    # Added mass (expected NEGATIVE)
    X_udot: float = -18.0
    Y_vdot: float = -22.584
    Z_wdot: float = -22.9775
    K_pdot: float = -0.079
    M_qdot: float = -0.26
    N_rdot: float = -0.286

    # Buoyancy and mass
    B: float = 17.2 * 9.81   # [N]
    m_val: float = 17.2      # [kg]

def _as_vec(x: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != 6:
        raise ValueError(f"{name} must be length-6, got shape {arr.shape}")
    return arr

def mass_matrix(params: ROVParams) -> np.ndarray:
    """M = M_RB + M_A (NED)."""
    m = params.m_val
    xg, yg, zg = params.x_cg, params.y_cg, params.z_cg

    Mrb = np.zeros((6, 6), dtype=float)
    Mrb[0, 0] = m
    Mrb[1, 1] = m
    Mrb[2, 2] = m

    Mrb[0, 4] =  m * zg
    Mrb[0, 5] = -m * yg
    Mrb[1, 3] = -m * zg
    Mrb[1, 5] =  m * xg
    Mrb[2, 3] =  m * yg
    Mrb[2, 4] = -m * xg

    Mrb[3, 1] = -m * zg
    Mrb[3, 2] =  m * yg
    Mrb[3, 3] = params.I_xx

    Mrb[4, 0] =  m * zg
    Mrb[4, 2] = -m * xg
    Mrb[4, 4] = params.I_yy

    Mrb[5, 0] = -m * yg
    Mrb[5, 1] =  m * xg
    Mrb[5, 5] = params.I_zz

    # Added mass uses the *_udot names
    Ma = -np.diag([
        params.X_udot, params.Y_vdot, params.Z_wdot,
        params.K_pdot, params.M_qdot, params.N_rdot
    ])

    return Mrb + Ma

def coriolis_matrix(nu: Iterable[float], params: ROVParams) -> np.ndarray:
    """C = C_RB + C_A (NED)."""
    u, v, w, p, q, r = _as_vec(nu, "nu")
    m = params.m_val
    Ixx, Iyy, Izz = params.I_xx, params.I_yy, params.I_zz

    Crb = np.zeros((6, 6), dtype=float)
    Crb[0, 4] =  m * w
    Crb[0, 5] = -m * v
    Crb[1, 3] = -m * w
    Crb[1, 5] =  m * u
    Crb[2, 3] =  m * v
    Crb[2, 4] = -m * u

    Crb[3, 1] = -m * w
    Crb[3, 2] =  m * v
    Crb[3, 4] =  Izz * r
    Crb[3, 5] = -Iyy * q

    Crb[4, 0] =  m * w
    Crb[4, 2] = -m * u
    Crb[4, 3] = -Izz * r
    Crb[4, 5] =  Ixx * p

    Crb[5, 0] = -m * v
    Crb[5, 1] =  m * u
    Crb[5, 3] =  Iyy * q
    Crb[5, 4] = -Ixx * p

    # Added-mass Coriolis uses *_udot names
    a1 = -params.X_udot * u
    a2 = -params.Y_vdot * v
    a3 = -params.Z_wdot * w
    b1 = -params.K_pdot * p
    b2 = -params.M_qdot * q
    b3 = -params.N_rdot * r

    Ca = np.zeros((6, 6), dtype=float)
    Ca[0, 4] = -a3
    Ca[0, 5] =  a2
    Ca[1, 3] =  a3
    Ca[1, 5] = -a1
    Ca[2, 3] = -a2
    Ca[2, 4] =  a1

    Ca[3, 1] = -a3
    Ca[3, 2] =  a2
    Ca[3, 4] = -b3
    Ca[3, 5] =  b2

    Ca[4, 0] =  a3
    Ca[4, 2] = -a1
    Ca[4, 3] =  b3
    Ca[4, 5] = -b1

    Ca[5, 0] = -a2
    Ca[5, 1] =  a1
    Ca[5, 3] = -b2
    Ca[5, 4] =  b1

    return Crb + Ca

def damping_matrix(nu: Iterable[float], params: ROVParams) -> np.ndarray:
    """Diagonal D(ν) = diag([X_u+X_uu|u|, ..., N_r+N_rr|r|])."""
    u, v, w, p, q, r = _as_vec(nu, "nu")
    diag = -np.array([
        params.X_u + params.X_uu * abs(u),
        params.Y_v + params.Y_vv * abs(v),
        params.Z_w + params.Z_ww * abs(w),
        params.K_p + params.K_pp * abs(p),
        params.M_q + params.M_qq * abs(q),
        params.N_r + params.N_rr * abs(r),
    ], dtype=float)
    return np.diag(diag)

def restoring_force(eta: Iterable[float], params: ROVParams) -> np.ndarray:
    """τ_g(η) following your PyTorch signs (NED)."""
    _, _, _, phi, theta, _ = _as_vec(eta, "eta")
    g = 9.81
    W = params.m_val * g
    B = params.B

    # CG/CB contributions (x_B,y_B used as zeros per your code)
    x_G, y_G, z_G = params.x_cg, params.y_cg, params.z_cg
    x_B, y_B, z_B = params.x_cb, params.y_cb, params.z_cb

    WB    = W - B
    xW_xB = x_G * W - x_B * B
    yW_yB = y_G * W - y_B * B
    zW_zB = z_G * W - z_B * B

    return np.array([
        WB * np.sin(theta),
        -WB * np.cos(theta) * np.sin(phi),
        -WB * np.cos(theta) * np.cos(phi),
        -yW_yB * np.cos(theta) * np.cos(phi) + zW_zB * np.cos(theta) * np.sin(phi),
        zW_zB * np.sin(theta) + xW_xB * np.cos(theta) * np.cos(phi),
        -xW_xB * np.cos(theta) * np.sin(phi) - yW_yB * np.sin(theta),
    ], dtype=float)

def compute_tau(nu_dot: Iterable[float], nu: Iterable[float], eta: Iterable[float], params: ROVParams) -> np.ndarray:
    """τ = M ν̇ + C(ν) ν + D(ν) ν + g(η)."""
    nu_dot = _as_vec(nu_dot, "nu_dot")
    nu     = _as_vec(nu, "nu")
    eta    = _as_vec(eta, "eta")

    M = mass_matrix(params)
    C = coriolis_matrix(nu, params)
    D = damping_matrix(nu, params)
    g_eta = restoring_force(eta, params)

    return (M @ nu_dot) + (C @ nu) + (D @ nu) + g_eta

# ---------- Example usage ----------
if __name__ == "__main__":
    P = ROVParams()

    # CSV line you provided:
    # 760198449.378018  0.129965 -0.045304 1.807869  0 0 -0.000068   0 -0 0.017546  0 0 -0.033333  0 0 -4.999269  0 -0 0   0 0 71.291442 0 0 6.374077  Z 0.4
    nu_dot_example = [0.0, 0.0, 3.337571, 0.0, 0.0, 0.0]
    nu_example     = [0.0, -0.0, 0.14893, 0.0, 0.0, 0.0]
    eta_example    = [0.0, 0.0, -4.999269, 0.0, -0.0, 0.0]

    tau = compute_tau(nu_dot_example, nu_example, eta_example, P)

    print("=== Parameters ===")
    for k, v in asdict(P).items():
        print(f"{k:10s}: {v:+.6f}")

    print("\nnu_dot:", np.array(nu_dot_example))
    print("nu    :", np.array(nu_example))
    print("eta   :", np.array(eta_example))

    print("\n--- Outputs ---")
    print("M @ nu_dot :", mass_matrix(P) @ np.asarray(nu_dot_example))
    print("C @ nu     :", coriolis_matrix(nu_example, P) @ np.asarray(nu_example))
    print("D @ nu     :", damping_matrix(nu_example, P) @ np.asarray(nu_example))
    print("g(eta)     :", restoring_force(eta_example, P))
    print("\nτ (total)  :", tau)
