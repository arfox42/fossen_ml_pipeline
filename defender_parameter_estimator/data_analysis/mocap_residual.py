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
# === 1. Select which DOF to analyze =========================
# ============================================================
ACTIVE_DOF = "X"   # options: "X", "Y", "Z", "K", "M", "N"


# ============================================================
# === 2. Load CSV ============================================
# ============================================================
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Coupled Maneuvers/csv_full_simulation_truth_teleop_circle.csv"

data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)
t = data[:, 0]
t_rel = t - t[0]

# body velocities
VEL_COL_MAP = {"X": 7, "Y": 8, "Z": 9, "K": 10, "M": 11, "N": 12}

# body forces / moments
TAU_COL_MAP = {"X": 19, "Y": 20, "Z": 21, "K": 22, "M": 23, "N": 24}

vel_meas = data[:, VEL_COL_MAP[ACTIVE_DOF]]
tau = data[:, TAU_COL_MAP[ACTIVE_DOF]]

# Euler angles if needed for later hydrostatics
phi   = data[:, 16]
theta = data[:, 17]
psi   = data[:, 18]


# ============================================================
# === 3. MLE parameters for all DOFs ==========================
# ============================================================
params = {

    # ---- LINEAR DOFs ----
    "X": {"mass": 17.2, "A": -17.438413619995117, "D1": -4.717586517333984,  "D2": -51.40349197387695,  "B": None},

    "Y": {"mass": 23.64, "A": -25.576709747314453, "D1": -35.01866912841797, "D2": -107.85147094726562,  "B": None},

    "Z": {
        "mass": 23.64,
        "A":  -54.148014068603516,
        "D1": -36.365638732910156,
        "D2": -130.41793823242188,
        "B":  238.98196411132812,   # MLE buoyancy
    },

    # ---- ROTATIONAL DOFs ----
    "K": {"mass": 1.2, "A": -0.3, "D1": -0.2,  "D2": -1.0, "B": None},
    "M": {"mass": 1.4, "A": -0.4, "D1": -0.25, "D2": -1.2, "B": None},
    "N": {"mass": 0.31836390495300293, "A": -0.3183645009994507, "D1": -2.8935487270355225,  "D2": -2.6945016384124756, "B": None},
}

p = params[ACTIVE_DOF]


# ============================================================
# === 4. Effective mass / inertia =============================
# ============================================================
m_eff = p["mass"] - p["A"]        # works for linear & rotational DOFs
W = p["mass"] * 9.81              # used only for Z
B = p["B"]


# ============================================================
# === 5. Restoring forces per DOF =============================
# ============================================================
# Default: no restoring term unless defined
g_i = np.zeros_like(t)

if ACTIVE_DOF == "Z":
    # heave restoring force
    g_val = W - B       # SIGN: positive Z down
    g_i[:] = g_val

elif ACTIVE_DOF in ("K", "M"):
    # rotational restoring would go here (roll/pitch moments),
    # but left as zero unless you want to activate it.
    g_i[:] = 0.0

# else X, Y, N → leave as zero


# ============================================================
# === 6. Forward integrate dynamics ===========================
# ============================================================

print("m_eff =", m_eff)
print("Range of tau:", np.nanmin(tau), np.nanmax(tau))
print("Range of vel_meas:", np.nanmin(vel_meas), np.nanmax(vel_meas))
print("Range of dt:", np.nanmin(np.diff(t)), np.nanmax(np.diff(t)))


vel_pred = np.zeros_like(vel_meas)
vel_pred[0] = vel_meas[0]

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    v_prev = vel_pred[i - 1]

    # FIXED SIGNS: p["D1"], p["D2"] are Fossen hydrodynamic derivatives
    drag_lin  = + p["D1"] * v_prev
    drag_quad = + p["D2"] * abs(v_prev) * v_prev

    acc = (tau[i - 1]
           - g_i[i - 1]
           + drag_lin
           + drag_quad) / m_eff

    vel_pred[i] = v_prev + acc * dt


# ============================================================
# === 7. Plot 3-panel diagnostic ===============================
# ============================================================
plt.figure(figsize=(14, 10))

# 1. Applied force/moment
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t_rel, tau, 'r-', label=f"{ACTIVE_DOF} Force/Moment")
ax1.set_ylabel(f"{ACTIVE_DOF} Force/Moment")
ax1.set_title(f"{ACTIVE_DOF} DOF Consistency Check")
ax1.grid(True)

# 2. measured vs predicted velocity
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(t_rel, vel_meas, label=f"{ACTIVE_DOF} measured", linewidth=2)
ax2.plot(t_rel, vel_pred, label=f"{ACTIVE_DOF} predicted", linewidth=2)
ax2.set_ylabel(f"{ACTIVE_DOF} velocity / rate")
ax2.set_ylim([-2, 2])
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
