#!/usr/bin/env python3
"""
plot_hmc_joint.py

Load saved HMC samples (.pt) and plot a joint posterior scatter
for (linear vs quadratic) terms of a selected DOF.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
LOAD_PATH  = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/hmc_outputs/hmc_surge_dropout_samples.pt"
ACTIVE_DOF = "X"   # "X","Y","Z","K","M","N"

# =========================
# DOF → parameter mapping
# =========================
name_map = {
    "X": ("X_dot_u", "X_u", "X_uu"),
    "Y": ("Y_dot_v", "Y_v", "Y_vv"),
    "Z": ("Z_dot_w", "Z_w", "Z_ww"),
    "K": ("K_dot_p", "K_p", "K_pp"),
    "M": ("M_dot_q", "M_q", "M_qq"),
    "N": ("N_dot_r", "N_r", "N_rr"),
}

if ACTIVE_DOF not in name_map:
    raise ValueError(f"ACTIVE_DOF must be one of {list(name_map.keys())}")

d_dot, d_lin, d_quad = name_map[ACTIVE_DOF]

# =========================
# Load saved samples
# =========================
data = torch.load(LOAD_PATH, map_location="cpu")
samples_tensor = data["samples"]     # (N, 26)
param_names    = data["param_names"] # list[str]

print(f"Loaded: {LOAD_PATH}")
print(f"samples_tensor shape: {tuple(samples_tensor.shape)}")

# =========================
# Extract samples for selected DOF
# =========================
idx = {n: i for i, n in enumerate(param_names)}
missing = [p for p in (d_dot, d_lin, d_quad) if p not in idx]
if missing:
    raise KeyError(f"Missing parameters in saved file: {missing}")

x_dot  = samples_tensor[:, idx[d_dot]].numpy()
x_lin  = samples_tensor[:, idx[d_lin]].numpy()
x_quad = samples_tensor[:, idx[d_quad]].numpy()

# =========================
# Correlations
# =========================
c_lq = np.corrcoef(x_lin, x_quad)[0, 1]
c_la = np.corrcoef(x_lin, x_dot)[0, 1]
c_qa = np.corrcoef(x_quad, x_dot)[0, 1]

print(f"\n=== {ACTIVE_DOF}-DOF posterior coupling ===")
print(f"corr({d_lin}, {d_quad}) = {c_lq:+.3f}")
print(f"corr({d_lin}, {d_dot})  = {c_la:+.3f}")
print(f"corr({d_quad}, {d_dot}) = {c_qa:+.3f}")

# =========================
# Plot: linear vs quadratic
# =========================
plt.figure(figsize=(6, 5))
plt.scatter(x_lin, x_quad, s=8, alpha=0.35)
plt.xlabel(d_lin)
plt.ylabel(d_quad)
plt.title(f"{d_lin} vs {d_quad}\n" f"corr = {c_lq:+.3f}  (DOF {ACTIVE_DOF})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
