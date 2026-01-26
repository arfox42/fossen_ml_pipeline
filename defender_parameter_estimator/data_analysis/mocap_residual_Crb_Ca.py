#!/usr/bin/env python3
"""
unified_dof_post_check.py
-------------------------------------
Forward-simulates a single DOF of the Defender using your MLE parameters
and compares predicted velocity vs measured velocity from MOCAP.

DOF selector supported: X, Y, Z, K, M, N
Generates a 3-panel diagnostic figure:
1) Applied force/moment τ_i
2) v_meas vs v_pred
3) residual = v_meas - v_pred
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# === 2. Load CSV (load ALL relevant columns explicitly) =====
# ============================================================
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Coupled Maneuvers/defender_data_teleop_circle_z_sin.csv"

data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)

# time
t     = data[:, 0]
t_rel = t - t[0]

# body-frame linear velocities (DVL)
u_meas = data[:, 7]    # surge
v_meas = data[:, 8]    # sway
w_meas = data[:, 9]    # heave

# body-frame angular velocities
p_meas = data[:, 10]   # roll rate
q_meas = data[:, 11]   # pitch rate
r_meas = data[:, 12]   # yaw rate

# mocap / estimator positions and orientations (if needed)
x     = data[:, 13]
y     = data[:, 14]
z     = data[:, 15]
phi   = data[:, 16]
theta = data[:, 17]
psi   = data[:, 18]

# body-frame forces / moments (from LUT → wrench)
tau_x = data[:, 19]
tau_y = data[:, 20]
tau_z = data[:, 21]
tau_K = data[:, 22]   # roll
tau_M = data[:, 23]   # pitch
tau_N = data[:, 24]   # yaw


# ============================================================
# === 3. Parameters for X DOF only ============================
# ============================================================

params_X = {
    "mass": 23.84,                      # physical mass
    "Ma":  -30.81025505065918,         # added mass X_dot_u (for m_eff)

    # Coriolis-only added-mass terms for X DOF:
    # Ca[0,4] = -a3,  a3 = Ca_Z_dot_w * w
    # Ca[0,5] =  a2,  a2 = Ca_Y_dot_v * v
    "Ca_Y_dot_v": -1.2956397533416748,
    "Ca_Z_dot_w": 23.104907989501953,

    "D1":  -9.046943664550781,         # linear drag
    "D2":  -51.31620407104492,         # quadratic drag
    "B":   None                        # not used for surge
}

p = params_X

# ============================================================
# === 4. Effective mass / inertia =============================
# ============================================================
m_eff = p["mass"] - p["Ma"]


# ============================================================
# === 6. Forward integrate dynamics (X DOF only) =============
# ============================================================

# allocate predicted surge velocity arrays
vel_pred_old = np.zeros_like(u_meas)
vel_pred_new = np.zeros_like(u_meas)

# initialize with measured initial surge velocity
vel_pred_old[0] = u_meas[0]
vel_pred_new[0] = u_meas[0]

for i in range(1, len(t)):
    dt = t[i] - t[i-1]

    # surge velocities (predicted)
    u_old = vel_pred_old[i-1]
    u_new = vel_pred_new[i-1]

    # other DOFs from measured data
    v_prev = v_meas[i-1]
    w_prev = w_meas[i-1]
    p_prev = p_meas[i-1]
    q_prev = q_meas[i-1]
    r_prev = r_meas[i-1]

    # === DRAG ===
    drag_old = p["D1"] * u_old + p["D2"] * abs(u_old) * u_old
    drag_new = p["D1"] * u_new + p["D2"] * abs(u_new) * u_new

    # === RIGID-BODY CORIOLIS (X-row) ===
    Cx_RB = p["mass"] * (w_prev * q_prev - v_prev * r_prev)

    # === ADDED-MASS CORIOLIS (OLD model: uses Ma) ===
    a2_old = p["Ma"] * v_prev
    a3_old = p["Ma"] * w_prev
    Cx_A_old = -a3_old * q_prev + a2_old * r_prev

    # === ADDED-MASS CORIOLIS (NEW model: uses Ca_Y_dot_v, Ca_Z_dot_w) ===
    a2_new = p["Ca_Y_dot_v"] * v_prev
    a3_new = p["Ca_Z_dot_w"] * w_prev
    Cx_A_new = -a3_new * q_prev + a2_new * r_prev

    # === ACCELERATIONS ===
    acc_old = (tau_x[i-1] + drag_old - Cx_RB - Cx_A_old) / m_eff
    acc_new = (tau_x[i-1] + drag_new - Cx_RB - Cx_A_new) / m_eff

    # === UPDATE VELOCITY ===
    vel_pred_old[i] = u_old + acc_old * dt
    vel_pred_new[i] = u_new + acc_new * dt




# ============================================================
# === 7. Plot 3-panel diagnostic ===============================
# ============================================================

plt.figure(figsize=(14, 12))

# ---------------------------
# 1. Applied surge force τ_x
# ---------------------------
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau_x, 'r-', linewidth=2, label="Applied Surge Force τ_x")
ax1.set_ylabel("τ_x [N]")
ax1.set_title("Surge (X) DOF Consistency Check")
ax1.grid(True)
ax1.legend()

# ---------------------------
# 2. measured vs predicted velocities
# ---------------------------
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t_rel, u_meas, 'y-', linewidth=2, label="Measured u")
ax2.plot(t_rel, vel_pred_old, 'b--', linewidth=2, label="Predicted u (OLD: Ma)")
ax2.plot(t_rel, vel_pred_new, 'g-.', linewidth=2, label="Predicted u (NEW: Ca)")
ax2.set_ylabel("u [m/s]")
ax2.grid(True)
ax2.legend()

# ---------------------------
# 3. residuals
# ---------------------------
res_old = u_meas - vel_pred_old
res_new = u_meas - vel_pred_new

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(t_rel, res_old, 'b--', linewidth=2, label="Residual (OLD: Ma)")
ax3.plot(t_rel, res_new, 'g-.', linewidth=2, label="Residual (NEW: Ca)")
ax3.set_ylabel("Residual u")
ax3.set_xlabel("Time [s]")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()


