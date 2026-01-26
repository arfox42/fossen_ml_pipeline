import numpy as np
import pandas as pd
import os

# --- parameters ---
m = 17.2
Z_w = -14.17
Z_ww = -155.8358
Z_wdot = -22.3775
M33 = m - Z_wdot  # 39.5775

# --- paths ---
SRC = os.path.join(os.path.dirname(__file__), "csv_full_z_run.csv")
DST = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_full_shifted_z_run.csv"

# --- load data ---
df = pd.read_csv(SRC, sep="\t")

# --- extract relevant columns ---
w_dot = df["w_dot"].to_numpy()
w     = df["w"].to_numpy()
Z_meas = df["Z"].to_numpy()

# --- model terms ---
Dz  = Z_w * w + Z_ww * np.abs(w) * w
lhs = (M33 * w_dot) - Dz   # model-predicted tau_Z (unshifted)

# --- find best integer lag between measured and model-predicted ---
def best_lag(a, b):
    A = a - a.mean(); B = b - b.mean()
    xcorr = np.correlate(A, B, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    return lags[np.argmax(xcorr)]

lag = best_lag(Z_meas, lhs)
print(f"Estimated best lag (tau_Z vs lhs): {lag} samples")

# --- shifting helper ---
def shift_array(arr, k):
    if k == 0:
        return arr
    out = np.empty_like(arr)
    if k > 0:
        out[k:] = arr[:-k]
        out[:k] = arr[0]
    else:
        out[:k] = arr[-k:]
        out[k:] = arr[-1]
    return out

# --- apply shift to all columns ---
df_shifted = df.copy()
for col in df.columns:
    df_shifted[col] = shift_array(df[col].to_numpy(), lag)

# --- recompute Z alignment diagnostics ---
Z_shift = shift_array(Z_meas, lag)
tau_model = lhs
residuals = tau_model - Z_shift

mean_abs = np.mean(np.abs(residuals))
p95_abs  = np.percentile(np.abs(residuals), 95)
r2 = 1 - np.sum(residuals**2) / np.sum((Z_shift - np.mean(Z_shift))**2)

print("\n=== τ_Z Residual Diagnostics ===")
print(f"Mean abs error:  {mean_abs:.3f} N")
print(f"95th percentile: {p95_abs:.3f} N")
print(f"R² coefficient:  {r2:.4f}")

# --- add diagnostic columns to saved CSV ---
df_shifted["tau_Z_model"] = tau_model
df_shifted["Z_shifted"]   = Z_shift
df_shifted["tau_Z_resid"] = residuals

df_shifted.to_csv(DST, sep="\t", index=False)
print(f"\n✅ Shifted and augmented CSV saved to:\n{DST}")
