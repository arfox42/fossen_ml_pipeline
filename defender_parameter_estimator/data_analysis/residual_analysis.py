#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "~/fossen_ml_pipeline/defender_parameter_estimator/csv_files/Z_Data/Sim_data/csv_full_truth_z_run_26OCT.csv"
T_START = 120.0          # search for first spike after this time [s]
HALF_WIDTH = 1.5        # seconds to show on either side of spike

# --- Model parameters (same as simulation) ---
m = 17.2
Z_w = -14.17
Z_ww = -155.8358
Z_wdot = -22.3775
M33 = m - Z_wdot  # = 39.5775

# ============================================================
# LOAD DATA
# ============================================================

headers = [
    "time",
    "u_dot","v_dot","w_dot","p_dot","q_dot","r_dot",
    "u","v","w","p","q","r",
    "x","y","z","phi","theta","psi",
    "X","Y","Z","K","M","N",
    "norm_dof","norm_value"
]

df = pd.read_csv(CSV_PATH, sep=r"\s+", header=None, names=headers)
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"]).reset_index(drop=True)
df["time"] -= df["time"].iloc[0]

t = df["time"].to_numpy()
w = df["w"].to_numpy()
w_dot = df["w_dot"].to_numpy()
Z_meas = df["Z"].to_numpy()

# ============================================================
# COMPUTE MODEL AND RESIDUALS
# ============================================================

Dz = Z_w * w + Z_ww * np.abs(w) * w
tau_Z_model = M33 * w_dot - Dz
resid = tau_Z_model - Z_meas
abs_resid = np.abs(resid)

mean_abs = np.mean(abs_resid)
p95_abs = np.percentile(abs_resid, 95)
r2 = 1 - np.sum(resid**2) / np.sum((Z_meas - np.mean(Z_meas))**2)
print(f"Global R² = {r2:.4f}, Mean |resid| = {mean_abs:.2f} N, 95th = {p95_abs:.2f} N")

# ============================================================
# FIND SPIKE AND DEFINE WINDOW
# ============================================================

def find_peak_after(t, abs_resid, t_start):
    idx_start = np.searchsorted(t, t_start, side="left")
    if idx_start >= len(t):
        raise ValueError("t_start beyond data range.")
    return idx_start + np.argmax(abs_resid[idx_start:])

def window_mask(t, center_t, half_width):
    return (t >= center_t - half_width) & (t <= center_t + half_width)

i_pk = find_peak_after(t, abs_resid, T_START)
t_pk = t[i_pk]
mask = window_mask(t, t_pk, HALF_WIDTH)

print(f"\n=== Zoom window around t ≈ {t_pk:.3f} s (±{HALF_WIDTH}s) ===")
print(f"Samples in window: {mask.sum()}")

# ============================================================
# COMPUTE TERMS IN WINDOW
# ============================================================

mass_term = M33 * w_dot
lin_term  = -Z_w * w
quad_term = -Z_ww * np.abs(w) * w
dz_term   = lin_term + quad_term
tau_model = tau_Z_model
tau_meas  = Z_meas
res_win   = resid[mask]

print(f"Residual stats: mean={np.mean(res_win):.3f}, "
      f"mean|.|={np.mean(np.abs(res_win)):.3f}, "
      f"max|.|={np.max(np.abs(res_win)):.3f} N")

# ============================================================
# CROSS-CORRELATION (residual vs w_dot)
# ============================================================

tw = t[mask]
wd = w_dot[mask] - np.mean(w_dot[mask])
rz = resid[mask] - np.mean(resid[mask])
corr = np.correlate(wd, rz, mode="full")
lags = np.arange(-len(wd) + 1, len(wd))
best_idx = np.argmax(np.abs(corr))
best_lag = lags[best_idx]
dt = np.median(np.diff(t))
print(f"Cross-corr lag (resid vs w_dot): {best_lag} samples ({best_lag*dt:.4f} s). "
      f"Sign>0 means resid lags acceleration.")

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# --- (1) Measured vs Model ---
axes[0].plot(t[mask], tau_meas[mask], label="Measured Z [N]", lw=1.6, color="tab:red")
axes[0].plot(t[mask], tau_model[mask], label="Model τ_Z [N]", lw=1.3, color="tab:green", alpha=0.85)
axes[0].axvline(t_pk, color="k", ls="--", lw=0.8)
axes[0].set_ylabel("Force [N]")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- (2) Term Breakdown ---
axes[1].plot(t[mask], mass_term[mask], label="M33·ẇ (inertial)", lw=1.3)
axes[1].plot(t[mask], lin_term[mask], label="-Z_w·w (linear)", lw=1.0)
axes[1].plot(t[mask], quad_term[mask], label="-Z_ww|w|w (quadratic)", lw=1.0)
axes[1].plot(t[mask], dz_term[mask], label="-(Z_w w + Z_ww|w|w)", lw=1.0, alpha=0.7)
axes[1].axvline(t_pk, color="k", ls="--", lw=0.8)
axes[1].set_ylabel("Force Components [N]")
axes[1].legend(ncol=2, fontsize=9)
axes[1].grid(alpha=0.3)

# --- (3) Residual + Acceleration ---
axes[2].plot(t[mask], resid[mask], label="Residual [N]", lw=1.2)
ax22 = axes[2].twinx()
ax22.plot(t[mask], w_dot[mask], "--", color="tab:orange", lw=1.0, label="ẇ [m/s²]")
axes[2].axvline(t_pk, color="k", ls="--", lw=0.8)
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Residual [N]")
ax22.set_ylabel("ẇ [m/s²]")
axes[2].grid(alpha=0.3)
h1, l1 = axes[2].get_legend_handles_labels()
h2, l2 = ax22.get_legend_handles_labels()
axes[2].legend(h1+h2, l1+l2, loc="best")

plt.suptitle(f"Zoom on Heave Residual Spike near t = {t_pk:.3f} s")
plt.tight_layout()
plt.show()

# --- Residual vs ẇ scatter ---
plt.figure(figsize=(6,4))
plt.scatter(w_dot[mask], resid[mask], s=8, alpha=0.6)
plt.xlabel("ẇ [m/s²]")
plt.ylabel("Residual [N]")
plt.title("Residual vs ẇ (zoom window)")
plt.grid(alpha=0.3)

# --- (4) Residual + Velocity over time ---
fig3, ax1 = plt.subplots(figsize=(10, 3.5), sharex=False)

# Residual (left y-axis)
ax1.plot(t[mask], resid[mask], color="tab:blue", lw=1.2, label="Residual [N]")
ax1.set_ylabel("Residual [N]", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(alpha=0.3)

# Velocity (right y-axis)
ax2 = ax1.twinx()
ax2.plot(t[mask], w[mask], color="tab:purple", lw=1.0, ls="--", label="Velocity w [m/s]")
ax2.set_ylabel("Velocity w [m/s]", color="tab:purple")
ax2.tick_params(axis="y", labelcolor="tab:purple")

ax1.axvline(t_pk, color="k", ls="--", lw=0.8)
ax1.set_xlabel("Time [s]")
plt.title("Residual vs Velocity (time-domain, zoom window)")

# Combine legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="best")

plt.tight_layout()
plt.show()



import numpy as np

def estimate_frac_lag(t, a, b):
    """
    Estimate fractional-sample lag between sequences a(t) and b(t).
    Returns lag_samples (positive => b lags a), lag_seconds.
    Uses xcorr peak + parabolic interpolation for sub-sample accuracy.
    """
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    corr = np.correlate(a0, b0, mode='full')
    lags = np.arange(-len(a0)+1, len(a0))
    k = np.argmax(np.abs(corr))
    # parabolic interpolation around k (guard edges)
    if 0 < k < len(corr)-1:
        y0, y1, y2 = corr[k-1], corr[k], corr[k+1]
        denom = (y0 - 2*y1 + y2)
        delta = 0.0 if denom == 0 else 0.5*(y0 - y2)/denom  # sub-sample offset
    else:
        delta = 0.0
    k_star = lags[k] + delta
    dt = np.median(np.diff(t))
    return k_star, k_star*dt

def frac_shift(x, shift_samples):
    """
    Linear fractional delay: y[n] = x[n - shift_samples]
    Positive shift => move signal to the RIGHT (delay).
    """
    n = np.arange(len(x))
    xi = n - shift_samples
    i0 = np.floor(xi).astype(int)
    a = xi - i0
    y = np.full_like(x, np.nan, dtype=float)
    valid = (i0 >= 0) & (i0+1 < len(x))
    y[valid] = (1-a[valid])*x[i0[valid]] + a[valid]*x[i0[valid]+1]
    return y

# ==== Use inside your zoom section ====
# We want τ_Z (measured) and M33*w_dot to be synchronous.
# Estimate lag between τ_Z_meas and w_dot in the zoom window:
wd_win = w_dot[mask]
tau_win = Z_meas[mask]

lag_samp, lag_sec = estimate_frac_lag(t[mask], tau_win, wd_win)
print(f"[Lag] τ_Z vs ẇ (window): {lag_samp:.3f} samples (~{lag_sec*1e3:.1f} ms). "
      "Positive => ẇ lags τ_Z.")

# Apply fractional shift to τ_Z so it lines up with ẇ:
Z_meas_aligned_frac = Z_meas.copy()
Z_meas_aligned_frac[mask] = frac_shift(Z_meas[mask], +lag_samp)  # delay τ by estimated lag

# Recompute residuals on window after fractional alignment:
Dz_win = Z_w*w[mask] + Z_ww*np.abs(w[mask])*w[mask]
tau_model_win = (M33*wd_win - Dz_win)
res_win_before = (M33*wd_win - Dz_win) - Z_meas[mask]
res_win_after  = (M33*wd_win - Dz_win) - Z_meas_aligned_frac[mask]

def rms(x): return np.sqrt(np.nanmean(x**2))
print(f"RMS residual (window) before: {rms(res_win_before):.2f} N | after: {rms(res_win_after):.2f} N")

# (Optional) replace your plotting residual series in the zoom figure with res_win_after

def corr(a,b):
    a=a-np.nanmean(a); b=b-np.nanmean(b);
    return np.nan_to_num(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
lin  = -Z_w*w[mask]
quad = -Z_ww*np.abs(w[mask])*w[mask]
iner = M33*w_dot[mask]
print("corr(res, inertial):", corr(res_win_before, iner))
print("corr(res, linear):  ", corr(res_win_before, lin))
print("corr(res, quad):    ", corr(res_win_before, quad))

dt = np.median(np.diff(t))
w_dot_simple = np.gradient(w, dt)

# 5-point Savitzky-Golay differentiation
from scipy.signal import savgol_filter
w_dot_savgol = savgol_filter(w, window_length=7, polyorder=3, deriv=1, delta=dt)

plt.plot(t, w_dot, label="Published accel (truth)")
plt.plot(t, w_dot_simple, label="np.gradient(DVL w)")
plt.plot(t, w_dot_savgol, label="Savitzky-Golay d/dt(w)", ls="--")
plt.legend(); plt.grid(True)
plt.show()

mask = np.abs(w_dot) > 0.2
gain = np.sum(w_dot_savgol[mask] * w_dot[mask]) / np.sum(w_dot[mask]**2)
print(f"Effective DVL-to-true acceleration gain ≈ {gain:.3f}")
