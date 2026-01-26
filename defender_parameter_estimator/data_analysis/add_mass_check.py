import numpy as np

# columns per your logger:
# time, u_dot,v_dot,w_dot,p_dot,q_dot,r_dot, u,v,w,p,q,r, x,y,z,phi,theta,psi, X,Y,Z,K,M,N, norm_dof,norm_val

data = np.loadtxt("rov_log_20251009_090900.csv", delimiter="\t", skiprows=1, dtype=float, usecols=range(0,24))

u_dot = data[:, 1]
u     = data[:, 7]
X     = data[:, 19]

m     = 17.2

# try with SIM defaults (or swap in learned values to test sensitivity)
X_u   = -4.66
X_uu  = -51.5

# build D11(u)
D11 = (-X_u) + (-X_uu)*np.abs(u)   # note: -X_u and -X_uu are positive if X_u, X_uu are negative

# filter out tiny udot to avoid division blowups
mask = np.abs(u_dot) > 1e-3
u_dot_f = u_dot[mask]
u_f     = u[mask]
X_f     = X[mask]
D11_f   = D11[mask]

# implied effective inertia and added mass
M11_eff     = (X_f - D11_f * u_f) / u_dot_f
Xu_dot_impl = m - M11_eff

print(f"Xu_dot implied: mean={Xu_dot_impl.mean():.3f}, std={Xu_dot_impl.std():.3f}, "
      f"median={np.median(Xu_dot_impl):.3f}, "
      f"p10={np.percentile(Xu_dot_impl,10):.3f}, p90={np.percentile(Xu_dot_impl,90):.3f}")

# Optional: check dependence on speed (are we compensating damping?)
bins = np.linspace(0, np.max(np.abs(u_f)), 8)
digit = np.digitize(np.abs(u_f), bins)
for b in range(1, len(bins)):
    vals = Xu_dot_impl[digit == b]
    if len(vals) > 20:
        print(f"|u| in [{bins[b-1]:.2f}, {bins[b]:.2f}] m/s -> Xu_dot mean={vals.mean():.2f}, std={vals.std():.2f}, n={len(vals)}")
