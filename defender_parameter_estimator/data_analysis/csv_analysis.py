import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# === USER SETTINGS ========================================
# ==========================================================
DOF = "X"   # choose: "X", "Y", "Z", "K", "M", or "N"

CSV_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/X_Data/Sim_data/csv_full_truth_x_run_26OCT.csv"
SEP = r"\s+"   # whitespace delimiter for your log format

# ==========================================================
# === PHYSICAL PARAMETERS (SIM TRUTH) ======================
# ==========================================================
params = {
    "X": {"m": 17.2, "d1": -4.66, "d2": -51.5, "a": -18.0},
    "Y": {"m": 17.2, "d1": -8.25, "d2": -102.006, "a": -22.584},
    "Z": {"m": 17.2, "d1": -14.17, "d2": -155.8358, "a": -22.3775},
    "K": {"m": 1.0,  "d1": -1.5,  "d2": -2.1,     "a": -0.079},
    "M": {"m": 1.0,  "d1": -2.9,  "d2": -14.6,    "a": -0.26},
    "N": {"m": 1.0,  "d1": -10.343, "d2": -8.8,   "a": -0.286},
}

# ==========================================================
# === COLUMN MAPPINGS PER DOF ==============================
# ==========================================================
col_map = {
    "X": {"vel": "u", "acc": "u_dot", "force": "X", "units": "N"},
    "Y": {"vel": "v", "acc": "v_dot", "force": "Y", "units": "N"},
    "Z": {"vel": "w", "acc": "w_dot", "force": "Z", "units": "N"},
    "K": {"vel": "p", "acc": "p_dot", "force": "K", "units": "N·m"},
    "M": {"vel": "q", "acc": "q_dot", "force": "M", "units": "N·m"},
    "N": {"vel": "r", "acc": "r_dot", "force": "N", "units": "N·m"},
}

# ==========================================================
# === LOAD CSV =============================================
# ==========================================================
headers = [
    "time",
    "u_dot","v_dot","w_dot","p_dot","q_dot","r_dot",
    "u","v","w","p","q","r",
    "x","y","z","phi","theta","psi",
    "X","Y","Z","K","M","N",
    "norm_dof","norm_value"
]

df = pd.read_csv(CSV_PATH, sep=SEP, header=None, names=headers)
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"]).reset_index(drop=True)
df["time"] = df["time"] - df["time"].iloc[0]

# ==========================================================
# === EXTRACT SIGNALS ======================================
# ==========================================================
c = col_map[DOF]
p = params[DOF]

t = df["time"].to_numpy()
vel = df[c["vel"]].to_numpy()
acc = df[c["acc"]].to_numpy()
tau_meas = df[c["force"]].to_numpy()

# ==========================================================
# === MODEL CALCULATION (generic: M, C_RB, g, D) ===========
# ==========================================================
M_eff = p["m"] - p["a"]                      # m - a  (or I - a for rotational DOFs)
D = p["d1"] * vel + p["d2"] * np.abs(vel) * vel

# Pull velocities/angles we need for C and g (arrays over time)
u = df["u"].to_numpy(); v = df["v"].to_numpy(); w = df["w"].to_numpy()
p_ang = df["p"].to_numpy(); q_ang = df["q"].to_numpy(); r_ang = df["r"].to_numpy()
phi = df["phi"].to_numpy(); theta = df["theta"].to_numpy()  # roll, pitch (rad)

# --- Coriolis/centripetal (rigid-body) C_RB * nu ---
# Forces:
tau_Cx = p["m"] * (v * r_ang - w * q_ang)       # X
tau_Cy = p["m"] * (w * p_ang - u * r_ang)       # Y
tau_Cz = p["m"] * (u * q_ang - v * p_ang)       # Z
# Moments (using diagonal inertias; your sim has Ixx=Iyy=Izz=1)
Ixx, Iyy, Izz = 1.0, 1.0, 1.0
tau_CK = (Izz - Iyy) * q_ang * r_ang            # K (roll)
tau_CM = (Ixx - Izz) * p_ang * r_ang            # M (pitch)
tau_CN = (Iyy - Ixx) * p_ang * q_ang            # N (yaw)

# Pick the component for the chosen DOF
C_map = {
    "X": tau_Cx, "Y": tau_Cy, "Z": tau_Cz,
    "K": tau_CK, "M": tau_CM, "N": tau_CN
}
tau_C = C_map[DOF]

# --- Restoring g(eta) ---
g0 = 9.81
W = 17.2 * g0          # [N]
B = 168.56             # [N]

# CG/CB (meters). You said z_cb = -0.05 (CB above CG, +Z is down)
z_g = 0.0
z_b = -0.05

# Build the common term zW_zB = z_cg*W - z_cb*B
zW_zB = z_g * W - z_b * B  # -> +8.428 N·m with your numbers

# Hydrostatic forces (same as you had)
tau_gX =  (W - B) * np.sin(theta)
tau_gY = -(W - B) * np.cos(theta) * np.sin(phi)
tau_gZ = -(W - B) * np.cos(theta) * np.cos(phi)

# Hydrostatic moments (small-angle; ignoring x/y offsets)
# >>> FIXED: use sin(phi) for K, sin(theta) for M, correct signs <<<
tau_gK =  zW_zB * np.sin(phi)        # roll restoring
tau_gM = zW_zB * np.sin(theta)      # pitch restoring
tau_gN =  0.0

g_map = {"X": tau_gX, "Y": tau_gY, "Z": tau_gZ,
         "K": tau_gK, "M": tau_gM, "N": tau_gN}
tau_g = g_map[DOF]



# --- Total modeled wrench component (this DOF) ---
tau_model = M_eff * acc - D + tau_C + tau_g


# ==========================================================
# === RESIDUALS & METRICS ==================================
# ==========================================================
resid = tau_model - tau_meas
abs_resid = np.abs(resid)

mean_abs = np.mean(abs_resid)
p95_abs = np.percentile(abs_resid, 95)
r2 = 1 - np.sum(resid**2) / np.sum((tau_meas - np.mean(tau_meas))**2)

mask = np.abs(acc) < 0.05  # approximate steady-state mask
r2_ss = 1 - np.sum(resid[mask]**2) / np.sum((tau_meas[mask] - np.mean(tau_meas[mask]))**2)

print(f"\n=== {DOF}-AXIS MODEL DIAGNOSTICS ===")
print(f"Mean abs error : {mean_abs:.2f} {c['units']}")
print(f"95th percentile: {p95_abs:.2f} {c['units']}")
print(f"R² (full)      : {r2:.4f}")
print(f"R² (steady)    : {r2_ss:.4f}\n")

# ==========================================================
# === PLOTS ================================================
# ==========================================================
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

ax1.plot(t, tau_meas, color="tab:red", label=f"Measured {DOF} [{c['units']}]", linewidth=1.5)
ax1.plot(t, tau_model, color="tab:green", label=f"Model {DOF} [{c['units']}]", linewidth=1.2, alpha=0.8)
ax1.set_ylabel(f"{DOF} Force/Moment [{c['units']}]", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2.plot(t, abs_resid, color="tab:blue", label="|Residual|", linewidth=1.0, alpha=0.7)
ax2.axhline(p95_abs, color="tab:blue", linestyle="--", linewidth=0.8, label="95th percentile")
ax2.set_ylabel("Residual magnitude", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

high_err = abs_resid > p95_abs
ax1.fill_between(t, np.min(tau_meas), np.max(tau_meas),
                 where=high_err, color="orange", alpha=0.2,
                 label=">95th% Error Region")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title(f"{DOF} Model vs Measured (Residual Diagnostics)")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()

# --- Residual vs velocity scatter ---
plt.figure(figsize=(6,4))
plt.scatter(vel, abs_resid, s=6, alpha=0.5)
plt.xlabel(f"{DOF}-axis velocity [{c['vel']}]")
plt.ylabel(f"|τ_{DOF} residual| [{c['units']}]")
plt.title(f"{DOF} Residual Magnitude vs Velocity")
plt.grid(True)
plt.tight_layout()
plt.show()
