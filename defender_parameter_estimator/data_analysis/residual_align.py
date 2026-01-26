#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
CSV_PATH   = "~/fossen_ml_pipeline/defender_parameter_estimator/csv_files/csv_full_long_z.csv"
T_START    = 50.0     # look for first big spike after this time [s]
HALF_WIDTH = 1.5      # zoom window half-width [s]

# Heave model params (your values)
m = 17.2
Z_w = -14.17
Z_ww = -155.8358
Z_wdot = -22.3775
M33 = m - Z_wdot  # 39.5775

# Lag application settings
#   method="integer": use integer-sample shift (robust/simple)
#   method="fractional": linear-interp for sub-sample alignment
APPLY_LAG = True
LAG_METHOD = "integer"   # "integer" or "fractional"
# ===================================================

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

dt = np.median(np.diff(t))

# ---- model (unshifted) ----
Dz = Z_w * w + Z_ww * np.abs(w) * w
tau_Z_model = M33 * w_dot - Dz
resid = tau_Z_model - Z_meas
abs_resid = np.abs(resid)

def r2(yhat, y):
    return 1 - np.sum((yhat - y)**2) / np.sum((y - np.mean(y))**2)

print(f"GLOBAL (no lag): R^2={r2(tau_Z_model, Z_meas):.4f}, "
      f"mean|resid|={np.mean(abs_resid):.2f} N, p95={np.percentile(abs_resid,95):.2f} N")

# ===================== find spike & window =====================
def find_peak_after(tt, metric, t_start):
    i0 = np.searchsorted(tt, t_start, side="left")
    i = i0 + np.argmax(metric[i0:])
    return i, tt[i]

i_pk, t_pk = find_peak_after(t, abs_resid, T_START)
mask = (t >= t_pk - HALF_WIDTH) & (t <= t_pk + HALF_WIDTH)
print(f"Zoom at t≈{t_pk:.3f}s, window samples={mask.sum()} (dt≈{dt*1000:.1f} ms)")

# ===================== lag estimation =====================
def best_lag_samples(sig_a, sig_b):
    """
    Returns lag k (in samples) maximizing |corr|,
    where positive k means: b lags a (b occurs later).
    """
    a = sig_a - np.mean(sig_a)
    b = sig_b - np.mean(sig_b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a)+1, len(a))
    k = lags[np.argmax(np.abs(corr))]
    return k

# Focus on dynamic regions to avoid flat segments dominating correlation
dyn_mask_global = (np.abs(w_dot) > 0.05) | (np.abs(np.diff(np.r_[Z_meas[0], Z_meas]))/dt > 5)  # crude
dyn_mask_global[:5] = dyn_mask_global[-5:] = True  # protect edges minimally

k_global = best_lag_samples(Z_meas[dyn_mask_global], tau_Z_model[dyn_mask_global])
k_window = best_lag_samples(Z_meas[mask],           tau_Z_model[mask])
print(f"Estimated lag (global): {k_global} samples ({k_global*dt:.4f} s). "
      f"Sign>0: model lags Z_meas")
print(f"Estimated lag (window): {k_window} samples ({k_window*dt:.4f} s).")

k_use = int(k_window) if abs(k_window) < 3 else int(k_global)

# ===================== apply lag (optional) =====================
def shift_integer(x, k, fill=np.nan):
    if k == 0: return x.copy()
    y = np.empty_like(x)
    y[:] = fill
    if k > 0:      # y[t] = x[t-k] ⇒ shift right (x lags)
        y[k:] = x[:-k]
    else:          # k < 0: shift left
        y[:k] = x[-k:]
    return y

def shift_fractional(x, k_samples):
    if np.isclose(k_samples, 0): return x.copy()
    # linear interpolation
    idx = np.arange(len(x)) - k_samples
    i0 = np.floor(idx).astype(int)
    alpha = idx - i0
    y = np.full_like(x, np.nan, dtype=float)
    valid = (i0 >= 0) & (i0+1 < len(x))
    y[valid] = (1 - alpha[valid]) * x[i0[valid]] + alpha[valid] * x[i0[valid] + 1]
    return y

# Manual tuning factor (in samples)
EXTRA_SHIFT_SAMPLES = -4  # negative moves measured further right (delays it)


if APPLY_LAG:
    # If Z_meas leads and model lags (your plot), we should delay Z_meas
    # OR advance the model. We'll delay Z_meas by k_use to align with model.
    # Delay Z_meas by the estimated lag (invert sign so positive delay shifts right)
    if LAG_METHOD == "integer":
        Z_meas_aligned = shift_integer(Z_meas, -(k_use + EXTRA_SHIFT_SAMPLES))
    else:
        Z_meas_aligned = shift_fractional(Z_meas, -(k_window + EXTRA_SHIFT_SAMPLES))

    # recompute metrics ignoring NaNs introduced by shifting
    valid = np.isfinite(Z_meas_aligned) & np.isfinite(tau_Z_model)
    resid_aligned = tau_Z_model[valid] - Z_meas_aligned[valid]
    print(f"\nAFTER ALIGN (k={k_use} samples ≈ {k_use*dt:.4f}s using {LAG_METHOD}): "
          f"R^2={r2(tau_Z_model[valid], Z_meas_aligned[valid]):.4f}, "
          f"mean|resid|={np.nanmean(np.abs(resid_aligned)):.2f} N, "
          f"p95={np.nanpercentile(np.abs(resid_aligned),95):.2f} N")
else:
    Z_meas_aligned = Z_meas
    resid_aligned = resid
    valid = np.ones_like(Z_meas, dtype=bool)

# === Ensure all arrays and mask have matching length after shift ===
min_len = min(
    len(t),
    len(Z_meas_aligned),
    len(tau_Z_model),
    len(w),
    len(w_dot),
    len(resid_aligned)
)

# truncate *everything* consistently
t = t[:min_len]
Z_meas_aligned = Z_meas_aligned[:min_len]
tau_Z_model = tau_Z_model[:min_len]
w = w[:min_len]
w_dot = w_dot[:min_len]
resid_aligned = resid_aligned[:min_len]
valid = valid[:min_len]

# also trim the unaligned data for the 'before' plot
Z_meas = Z_meas[:min_len]
resid = resid[:min_len]


# rebuild mask based on the trimmed time vector
mask = (t >= t_pk - HALF_WIDTH) & (t <= t_pk + HALF_WIDTH)


# ===================== plots =====================
def panel(title, Zm, model, res, mask_plot, wdot=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # (1) forces
    axes[0].plot(t[mask_plot], Zm[mask_plot], label="Measured Z [N]", color="tab:red", lw=1.6)
    axes[0].plot(t[mask_plot], model[mask_plot], label="Model τ_Z [N]", color="tab:green", lw=1.3)
    axes[0].axvline(t_pk, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("Force [N]"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # (2) term breakdown
    mass_term = M33 * w_dot
    lin_term  = -Z_w * w
    quad_term = -Z_ww * np.abs(w) * w
    dz_term   = lin_term + quad_term
    axes[1].plot(t[mask_plot], mass_term[mask_plot], label="M33·ẇ", lw=1.2)
    axes[1].plot(t[mask_plot], lin_term[mask_plot],  label="-Z_w w", lw=1.0)
    axes[1].plot(t[mask_plot], quad_term[mask_plot], label="-Z_ww|w|w", lw=1.0)
    axes[1].plot(t[mask_plot], dz_term[mask_plot],   label="-(Z_w w + Z_ww|w|w)", lw=1.0, alpha=0.7)
    axes[1].axvline(t_pk, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("Components [N]"); axes[1].legend(ncol=2, fontsize=9); axes[1].grid(alpha=0.3)

    # (3) residual + accel
    axes[2].plot(t[mask_plot], res[mask_plot], label="Residual [N]", lw=1.2)
    ax2 = axes[2].twinx()
    ax2.plot(t[mask_plot], w_dot[mask_plot], "--", lw=1.0, label="ẇ [m/s²]", color="tab:orange")
    axes[2].axvline(t_pk, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("Time [s]"); axes[2].set_ylabel("Residual [N]"); ax2.set_ylabel("ẇ [m/s²]")
    h1,l1 = axes[2].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1+h2, l1+l2, loc="best"); axes[2].grid(alpha=0.3)
    plt.suptitle(title); plt.tight_layout(); plt.show()

# Before
panel("BEFORE alignment (zoom)", Z_meas, tau_Z_model, resid, mask)

# After
panel("AFTER alignment (zoom)", Z_meas_aligned, tau_Z_model, resid_aligned if APPLY_LAG else resid, mask)

# Scatter (after)
valid_zoom = valid & mask
plt.figure(figsize=(6,4))
plt.scatter(w_dot[valid_zoom], (tau_Z_model - Z_meas_aligned)[valid_zoom], s=8, alpha=0.6)
plt.xlabel("ẇ [m/s²]"); plt.ylabel("Residual [N]"); plt.title("Residual vs ẇ (zoom, aligned)")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
