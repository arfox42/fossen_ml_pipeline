#!/usr/bin/env python3
"""
fossen_residual_full.py
-----------------------
Forward-simulates the FULL 6-DOF Defender dynamics using your MLE parameters
and compares ONE selected DOF velocity vs measured velocity from MOCAP.

Uses full Fossen structure:
    M(ν) ν_dot + C(ν) ν + D(ν) ν + g(η) = τ

You pick ACTIVE_DOF ∈ { "X", "Y", "Z", "K", "M", "N" } and the script:
    1) Integrates ν forward in time from measured initial ν(0)
    2) Plots τ_i (applied force/moment in that DOF)
    3) Plots v_meas vs v_pred for that DOF
    4) Plots residual = v_meas - v_pred for that DOF
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# === 1. Select DOF + CSV ====================================
# ============================================================

ACTIVE_DOF = "N"   # "X", "Y", "Z", "K", "M", "N"

csv_path = (
    "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/"
    "csv_files/defender_data_teleop_circle.csv"
)

# DOF index mapping in ν = [u, v, w, p, q, r]
DOF_IDX = {"X": 0, "Y": 1, "Z": 2, "K": 3, "M": 4, "N": 5}

# Column mappings for your standard CSV
TIME_COL = 0
VEL_START_COL = 7   # u,v,w,p,q,r
POS_START_COL = 13  # x,y,z,phi,theta,psi
TAU_START_COL = 19  # X,Y,Z,K,M,N


# ============================================================
# === 2. Load CSV ============================================
# ============================================================

data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)

t = data[:, TIME_COL]
t_rel = t - t[0]

# measured body velocities (ν_meas)
nu_meas = data[:, VEL_START_COL : VEL_START_COL + 6]   # [u,v,w,p,q,r]

# pose (η) for g(η): x,y,z,phi,theta,psi
eta = data[:, POS_START_COL : POS_START_COL + 6]

# applied wrench τ (body frame)
tau = data[:, TAU_START_COL : TAU_START_COL + 6]

# Convenience handles
idx = DOF_IDX[ACTIVE_DOF]
vel_meas = nu_meas[:, idx]
tau_i = tau[:, idx]


# ============================================================
# === 3. Parameter block (Fossen model) ======================
# ============================================================
# Fill these from your MLE summary. Values below are EXAMPLES / placeholders.
# ------------------------------------------------------------------

# Rigid-body mass & inertia
m   = 23.84
Ixx = 0.016068289056420326   # <-- put your MLE I_xx
Iyy = 0.03671276941895485   # <-- put your MLE I_yy
Izz = 0.3186814486980438   # <-- put your MLE I_zz

# CG and CB (relative to body origin)
x_g, y_g, z_g = 0.0, 0.0, -0.0   # MLE CG
x_b, y_b, z_b = 0.0, 0.0, -0.028     # MLE CB (often x_b=y_b=0)

# Weight & buoyancy
W = m * 9.81
B = 232.98   # <-- from your Z-fit; update as needed

# Added mass (diagonal)
X_dot_u = -31.888933181762695
Y_dot_v = -60.0
Z_dot_w = -54.148014068603516
K_dot_p =  -0.01606844551861286
M_dot_q = -0.03671274706721306
N_dot_r = -0.3183645009994507

# Linear damping (diagonal)
X_u = -9.125594139099121
Y_v = -35.594122886657715
Z_w = -36.365638732910156
K_p = -0.7733964323997498
M_q = -1.6791133880615234
N_r = -2.8935487270355225

# Quadratic damping (diagonal)
X_uu = -51.10944747924805
Y_vv = -158.84963989257812
Z_ww = -130.41793823242188
K_pp = -0.3401205539703369
M_qq = -2.0740668773651123
N_rr = -2.6945016384124756


# ============================================================
# === 4. Helpers: Skew, M_RB, M_A, C_RB, C_A, D, g ===========
# ============================================================

def skew(v):
    """Return skew-symmetric matrix S(v) such that S(v) w = v × w."""
    x, y, z = v
    return np.array([
        [ 0.0, -z,   y ],
        [  z,  0.0, -x ],
        [ -y,  x,  0.0]
    ])


def M_RB():
    """Rigid-body mass matrix matching PyTorch version EXACTLY."""
    MRB = np.zeros((6, 6))

    MRB[0, 0] = m
    MRB[1, 1] = m
    MRB[2, 2] = m

    MRB[0, 4] =  m * z_g
    MRB[0, 5] = -m * y_g

    MRB[1, 3] = -m * z_g
    MRB[1, 5] =  m * x_g

    MRB[2, 3] =  m * y_g
    MRB[2, 4] = -m * x_g

    MRB[3, 1] = -m * z_g
    MRB[3, 2] =  m * y_g
    MRB[3, 3] = Ixx

    MRB[4, 0] =  m * z_g
    MRB[4, 2] = -m * x_g
    MRB[4, 4] = Iyy

    MRB[5, 0] = -m * y_g
    MRB[5, 1] =  m * x_g
    MRB[5, 5] = Izz

    return MRB



def M_A():
    """Added mass matrix, diagonal."""
    MA = np.zeros((6, 6))
    MA[0, 0] = -X_dot_u
    MA[1, 1] = -Y_dot_v
    MA[2, 2] = -Z_dot_w
    MA[3, 3] = -K_dot_p
    MA[4, 4] = -M_dot_q
    MA[5, 5] = -N_dot_r
    return MA


def C_RB(nu):
    """Rigid-body Coriolis matching PyTorch implementation exactly (CG ignored)."""
    u, v, w, p, q, r = nu
    C = np.zeros((6, 6))

    C[0, 4] =  m * w
    C[0, 5] = -m * v
    C[1, 3] = -m * w
    C[1, 5] =  m * u
    C[2, 3] =  m * v
    C[2, 4] = -m * u

    C[3, 1] = -m * w
    C[3, 2] =  m * v
    C[3, 4] =  Izz * r
    C[3, 5] = -Iyy * q

    C[4, 0] =  m * w
    C[4, 2] = -m * u
    C[4, 3] = -Izz * r
    C[4, 5] =  Ixx * p

    C[5, 0] = -m * v
    C[5, 1] =  m * u
    C[5, 3] =  Iyy * q
    C[5, 4] = -Ixx * p

    return C



def C_A(nu):
    """Added-mass Coriolis matching PyTorch implementation EXACTLY."""
    u, v, w, p, q, r = nu

    a1 = X_dot_u * u
    a2 = Y_dot_v * v
    a3 = Z_dot_w * w
    b1 = K_dot_p * p
    b2 = M_dot_q * q
    b3 = N_dot_r * r

    CA = np.zeros((6, 6))

    CA[0, 4] = -a3
    CA[0, 5] =  a2
    CA[1, 3] =  a3
    CA[1, 5] = -a1
    CA[2, 3] = -a2
    CA[2, 4] =  a1

    CA[3, 1] = -a3
    CA[3, 2] =  a2
    CA[3, 4] = -b3
    CA[3, 5] =  b2

    CA[4, 0] =  a3
    CA[4, 2] = -a1
    CA[4, 3] =  b3
    CA[4, 5] = -b1

    CA[5, 0] = -a2
    CA[5, 1] =  a1
    CA[5, 3] = -b2
    CA[5, 4] =  b1

    return CA



def D_mat(nu):
    """Diagonal damping EXACTLY matching PyTorch D(ν)."""
    u, v, w, p, q, r = nu

    linear = np.array([X_u, Y_v, Z_w, K_p, M_q, N_r])
    quad   = np.array([
        X_uu * abs(u),
        Y_vv * abs(v),
        Z_ww * abs(w),
        K_pp * abs(p),
        M_qq * abs(q),
        N_rr * abs(r)
    ])

    diag = linear + quad
    D = -np.diag(diag)  # PyTorch multiplies by -1

    return D



def g_vec(eta_i):
    """Restoring forces EXACTLY matching PyTorch."""
    phi  = eta_i[3]
    theta = eta_i[4]

    sphi = np.sin(phi)
    cphi = np.cos(phi)
    sthe = np.sin(theta)
    cthe = np.cos(theta)

    WB = W - B

    # weight-buoyancy terms
    xW_xB = x_g * W - 0.0 * B
    yW_yB = y_g * W - 0.0 * B
    zW_zB = z_g * W - z_b * B

    g = np.zeros(6)

    # Forces
    g[0] = WB * sthe
    g[1] = -WB * cthe * sphi
    g[2] = -WB * cthe * cphi

    # Moments
    g[3] = -yW_yB * cthe * cphi + zW_zB * cthe * sphi
    g[4] =  zW_zB * sthe + xW_xB * cthe * cphi
    g[5] = -xW_xB * cthe * sphi - yW_yB * sthe

    return g



# Precompute constant mass matrix
M_tot = M_RB() + M_A()
M_inv = np.linalg.inv(M_tot)


# ============================================================
# === 5. Forward integrate full 6-DOF dynamics ===============
# ============================================================

print("Using full Fossen model for residual check")
print("M_tot:\n", M_tot)

N = len(t)
nu_pred = np.zeros_like(nu_meas)
nu_pred[0, :] = nu_meas[0, :]  # start from measured initial ν

for i in range(1, N):
    dt = t[i] - t[i - 1]

    # clamp dt to avoid blow-ups on large gaps
    if dt <= 0 or dt > 0.05:
        dt = 0.01

    nu_prev = nu_pred[i - 1, :]
    eta_prev = eta[i - 1, :]

    C = C_RB(nu_prev) + C_A(nu_prev)
    D = D_mat(nu_prev)
    g = g_vec(eta_prev)
    tau_i_full = tau[i - 1, :]

    if i < 5 or i % 500 == 0:  # print first few + every 500 steps
        print(f"\n---- DEBUG i={i} ----")
        print("nu_prev =", nu_prev)
        print("max |nu_prev| =", np.nanmax(np.abs(nu_prev)))
        print("C:\n", C)
        print("max |C| =", np.nanmax(np.abs(C)))
        print("D diag =", np.diag(D))
        print("g =", g)
        print("tau =", tau_i_full)
        print("dt =", dt)

    # nu_dot = M^{-1}(tau - C nu - D nu - g)
    rhs = tau_i_full - C @ nu_prev - D @ nu_prev - g
    nu_dot = M_inv @ rhs

    nu_pred[i, :] = nu_prev + nu_dot * dt


vel_pred = nu_pred[:, idx]  # selected DOF


# ============================================================
# === 6. Plot diagnostics ====================================
# ============================================================

plt.figure(figsize=(14, 10))

# 1. Applied force/moment in selected DOF
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau_i, 'r-', label=f"{ACTIVE_DOF} Force/Moment")
ax1.set_ylabel(f"{ACTIVE_DOF} Force/Moment")
ax1.set_title(f"{ACTIVE_DOF} DOF Full-Fossen Consistency Check")
ax1.grid(True)

# 2. measured vs predicted velocity
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t_rel, vel_meas, label=f"{ACTIVE_DOF} measured", linewidth=2)
ax2.plot(t_rel, vel_pred, label=f"{ACTIVE_DOF} predicted", linewidth=2)
ax2.set_ylabel(f"{ACTIVE_DOF} velocity / rate")
ax2.set_ylim([-2, 2])   # tweak as needed
ax2.legend()
ax2.grid(True)

# 3. residual
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(t_rel, vel_meas - vel_pred, 'k-', label="Residual")
ax3.set_ylabel("Residual")
ax3.set_xlabel("Time [s]")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
