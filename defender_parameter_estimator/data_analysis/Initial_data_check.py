import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load Defender CSV
# ============================================================
csv_path = "/home/andrew/fossen_ml_pipeline/defender_parameter_estimator/csv_files/N_Data/Tank_data/defender_data_n_run_1_only_mocap_data_savgol.csv"

data = np.genfromtxt(csv_path, delimiter="\t", skip_header=1)


# Column mapping:
# time, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot,
# u, v, w, p, q, r,
# x, y, z, phi, theta, psi,
# X, Y, Z, K, M, N

time   = data[:, 0]


# Linear accelerations (IMU)
u_dot  = data[:, 1]

v_dot  = data[:, 2]
w_dot  = data[:, 3]

# Angular accelerations (from mocap diff)
p_dot  = data[:, 4]
q_dot  = data[:, 5]
r_dot  = data[:, 6]

# Velocities (mocap)
u      = data[:, 7]
v      = data[:, 8]
w      = data[:, 9]

# Angular velocities
p      = data[:, 10]
q      = data[:, 11]
r      = data[:, 12]

# Forces and torques
X      = data[:, 19]
Y      = data[:, 20]
Z      = data[:, 21]
K      = data[:, 22]
M      = data[:, 23]
N      = data[:, 24]


# ============================================================
# Helper: Dual-axis plot
# ============================================================
def dual_plot(time, force, qty, force_label, qty_label, title):
    plt.figure(figsize=(10,5))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Force / moment
    ax1.plot(time, force, label=force_label, color='tab:red', linewidth=2)
    ax1.set_ylabel(force_label, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Quantity
    ax2.plot(time, qty, label=qty_label, color='tab:blue', linewidth=2)
    ax2.set_ylabel(qty_label, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax1.set_xlabel("Time [s]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# Achieved Force / Moment Limits
# ============================================================

def finite(x):
    return x[np.isfinite(x)]

def force_limits_robust(name, x):
    x = finite(x)
    if x.size == 0:
        print(f"[{name}] no finite samples")
        return
    lo, hi = np.percentile(x, [1, 99])
    print(f"{name:>12s} | p01 = {lo: .3f} | p99 = {hi: .3f}")

print("\n=== Achieved Force / Moment Limits (Robust) ===")
for nm, arr in [("X [N]", X), ("Y [N]", Y), ("Z [N]", Z), ("K [Nm]", K), ("M [Nm]", M), ("N [Nm]", N)]:
    force_limits_robust(nm, arr)




# ============================================================
# Linear DOF: Force vs Linear Acceleration
# ============================================================

dual_plot(time, X, u_dot, "Surge Force X [N]", "u̇ [m/s²]",
          "Surge: X vs u̇")

dual_plot(time, Y, v_dot, "Sway Force Y [N]", "v̇ [m/s²]",
          "Sway: Y vs v̇")

dual_plot(time, Z, w_dot, "Heave Force Z [N]", "ẇ [m/s²]",
          "Heave: Z vs ẇ")


# ============================================================
# Linear DOF: Force vs Velocity
# ============================================================

dual_plot(time, X, u, "Surge Force X [N]", "u [m/s]",
          "Surge: X vs u")

dual_plot(time, Y, v, "Sway Force Y [N]", "v [m/s]",
          "Sway: Y vs v")

dual_plot(time, Z, w, "Heave Force Z [N]", "w [m/s]",
          "Heave: Z vs w")


# ============================================================
# Rotational DOF: Torque vs Angular Velocity
# ============================================================

dual_plot(time, K, p, "Roll Torque K [Nm]", "Roll Rate p [rad/s]",
          "Roll: K vs p")

dual_plot(time, M, q, "Pitch Torque M [Nm]", "Pitch Rate q [rad/s]",
          "Pitch: M vs q")

dual_plot(time, N, r, "Yaw Torque N [Nm]", "Yaw Rate r [rad/s]",
          "Yaw: N vs r")


# ============================================================
# NEW: Torque vs Angular Acceleration
# ============================================================

dual_plot(time, K, p_dot, "Roll Torque K [Nm]", "Roll Accel ṗ [rad/s²]",
          "Roll: K vs ṗ")

dual_plot(time, M, q_dot, "Pitch Torque M [Nm]", "Pitch Accel q̇ [rad/s²]",
          "Pitch: M vs q̇")

dual_plot(time, N, r_dot, "Yaw Torque N [Nm]", "Yaw Accel ṙ [rad/s²]",
          "Yaw: N vs ṙ")


# # ============================================================
# # Analysis: w_dot channel sanity stats
# # ============================================================
#
# def basic_stats(name, x):
#     x = finite(x)
#     if x.size == 0:
#         print(f"[{name}] no finite samples")
#         return
#     print(f"\n=== {name} stats ===")
#     print(f"N samples      : {x.size}")
#     print(f"mean           : {np.mean(x): .6f}")
#     print(f"std            : {np.std(x): .6f}")
#     print(f"min            : {np.min(x): .6f}")
#     print(f"max            : {np.max(x): .6f}")
#     print(f"p01 / p50 / p99: {np.percentile(x,1): .6f}  {np.percentile(x,50): .6f}  {np.percentile(x,99): .6f}")
#     print(f"mean |x|       : {np.mean(np.abs(x)): .6f}")
#
# # --- overall stats ---
# basic_stats("w_dot (overall)", w_dot)

# # ============================================================
# # Define "excited" windows using Z command
# # ============================================================
#
# Z_abs = np.abs(Z)
# Z_abs_f = finite(Z_abs)
#
# # Robust threshold: top 20% of |Z| counts as "excited"
# excited_percentile = 80  # you can change (e.g., 70, 85, 90)
# thr = np.percentile(Z_abs_f, excited_percentile)
#
# exc_mask = (Z_abs >= thr) & np.isfinite(w_dot) & np.isfinite(Z)
#
# w_dot_exc = w_dot[exc_mask]
# Z_exc = Z[exc_mask]
#
# print("\n=== Excitation definition ===")
# print(f"Using excited_percentile = {excited_percentile} (top {100-excited_percentile}% of |Z|)")
# print(f"|Z| threshold (N)         = {thr:.3f}")
# print(f"Excited samples           = {w_dot_exc.size} / {w_dot.size} ({100*w_dot_exc.size/max(1,w_dot.size):.1f}%)")
#
# # --- excited stats ---
# basic_stats("w_dot (excited)", w_dot_exc)
#
# # Optional: split excited into positive/negative heave force
# pos_mask = exc_mask & (Z > 0)
# neg_mask = exc_mask & (Z < 0)
#
# basic_stats("w_dot (excited, Z>0)", w_dot[pos_mask])
# basic_stats("w_dot (excited, Z<0)", w_dot[neg_mask])
#
# # Optional: show correlation between Z and w_dot during excited segments
# if w_dot_exc.size > 10:
#     corr = np.corrcoef(Z[exc_mask], w_dot_exc)[0, 1]
#     print(f"\nCorr(Z, w_dot) during excited: {corr:.3f}")
#
