#!/usr/bin/env python3
"""
spike_analyzer.py
-----------------
Analyzes residual spikes (|τ_model − τ_meas|) and decomposes the
Fossen LHS into Mν̇, −Dν, Cν, and g(η) terms.

Just set CSV_PATH and DOF below, then run:
    python3 spike_analyzer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# === USER SETTINGS ========================================
# ==========================================================
CSV_PATH = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/K_Data/Sim_data/csv_full_truth_k_test.csv"
DOF = "K"  # choose: "X", "Y", "Z", "K", "M", "N"
SEP = r"\s+"

# ==========================================================
# === VEHICLE PARAMETERS ===================================
# ==========================================================
params = {
    "X": {"m": 17.2, "d1": -4.66, "d2": -51.5, "a": -18.0},
    "Y": {"m": 17.2, "d1": -8.25, "d2": -102.006, "a": -22.584},
    "Z": {"m": 17.2, "d1": -14.17, "d2": -155.8358, "a": -22.3775},
    "K": {"m": 1.0,  "d1": -1.5,  "d2": -2.1,     "a": -0.079},
    "M": {"m": 1.0,  "d1": -2.9,  "d2": -14.6,    "a": -0.26},
    "N": {"m": 1.0,  "d1": -10.343, "d2": -8.8,   "a": -0.286},
}

col_map = {
    "X": {"vel": "u", "acc": "u_dot", "force": "X", "units": "N"},
    "Y": {"vel": "v", "acc": "v_dot", "force": "Y", "units": "N"},
    "Z": {"vel": "w", "acc": "w_dot", "force": "Z", "units": "N"},
    "K": {"vel": "p", "acc": "p_dot", "force": "K", "units": "N·m"},
    "M": {"vel": "q", "acc": "q_dot", "force": "M", "units": "N·m"},
    "N": {"vel": "r", "acc": "r_dot", "force": "N", "units": "N·m"},
}

# ==========================================================
# === LOAD DATA ============================================
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
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"])
df["time"] -= df["time"].iloc[0]

# ==========================================================
# === EXTRACT SIGNALS ======================================
# ==========================================================
c = col_map[DOF]
p = params[DOF]

t = df["time"].to_numpy()
vel = df[c["vel"]].to_numpy()
acc = df[c["acc"]].to_numpy()
tau_meas = df[c["force"]].to_numpy()

phi = df["phi"].to_numpy(); theta = df["theta"].to_numpy()
q_ang = df["q"].to_numpy(); r_ang = df["r"].to_numpy()
u = df["u"].to_numpy(); v = df["v"].to_numpy(); w = df["w"].to_numpy()

# ==========================================================
# === MODEL COMPONENTS =====================================
# ==========================================================
M_eff = p["m"] - p["a"]

# Damping term (negative coefficients → use −Dν form)
D_term = p["d1"]*vel + p["d2"]*np.abs(vel)*vel

Ixx=Iyy=Izz=1.0
tau_CK=(Izz-Iyy)*q_ang*r_ang
tau_CM=(Ixx-Izz)*vel*r_ang
tau_CN=(Iyy-Ixx)*vel*q_ang
C_map={"K":tau_CK,"M":tau_CM,"N":tau_CN}
tau_C = C_map.get(DOF, np.zeros_like(t))

# --- Restoring ---
g0=9.81; W=17.2*g0; B=168.56; z_g=0.0; z_b=-0.05
zW_zB = z_g*W - z_b*B
tau_gK =  zW_zB * np.sin(phi)
tau_gM = -zW_zB * np.sin(theta)
tau_gN = np.zeros_like(t)
g_map={"K":tau_gK,"M":tau_gM,"N":tau_gN}
tau_g = g_map.get(DOF, np.zeros_like(t))

# --- Full model (Fossen −Dν form) ---
tau_model = M_eff*acc - D_term + tau_C + tau_g
resid = tau_model - tau_meas
abs_resid = np.abs(resid)

# ==========================================================
# === SPIKE DETECTION ======================================
# ==========================================================
p95 = np.percentile(abs_resid, 95)
spike_mask = abs_resid > p95
spike_idxs = np.where(spike_mask)[0]

segments=[]
if len(spike_idxs)>0:
    start=spike_idxs[0]
    for i in range(1,len(spike_idxs)):
        if spike_idxs[i] != spike_idxs[i-1]+1:
            segments.append((start,spike_idxs[i-1]))
            start=spike_idxs[i]
    segments.append((start,spike_idxs[-1]))

print(f"Found {len(segments)} spike segments (>95th percentile)")

# ==========================================================
# === VISUALIZE SPIKES =====================================
# ==========================================================
for (s,e) in segments:
    pad=50
    i0=max(0,s-pad); i1=min(len(t)-1,e+pad)
    tslice=slice(i0,i1)

    plt.figure(figsize=(10,5))
    plt.title(f"{DOF}-axis spike window [{t[i0]:.2f}–{t[i1]:.2f}] s")
    plt.plot(t[tslice], tau_meas[tslice], "r", label="Measured τ")
    plt.plot(t[tslice], tau_model[tslice], "g", label="Model τ")
    plt.plot(t[tslice], resid[tslice], "b--", alpha=0.7, label="Residual")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel(f"{DOF} Force/Moment [{c['units']}]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Component decomposition at spike center
    mid = (s+e)//2
    terms = {
        "Mν̇": M_eff*acc[mid],
        "−Dν": -D_term[mid],
        "Cν": tau_C[mid],
        "g(η)": tau_g[mid],
        "τ_meas": tau_meas[mid],
    }

    print(f"\n=== Spike @ t={t[mid]:.3f}s ===")
    for k,v in terms.items():
        print(f"{k:8s}: {v:+.4f}")
    print(f"Residual : {(tau_model[mid]-tau_meas[mid]):+.4f}\n")
