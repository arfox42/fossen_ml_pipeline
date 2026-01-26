#!/usr/bin/env python3
"""
prior_added_mass_linear_and_rotational.py

Physics-informed (weakly-informative) priors for:

(1) Added mass (translational + rotational) via C_A scaling
(2) Quadratic drag (translational) via standard empirical drag law (Prestero-style)
(3) Quadratic angular drag (rotational) via a simple torque-drag scaling law

--------------------------------------------------------------------
(1) ADDED MASS (same as before)
  Translational:
    X_dot_u, Y_dot_v, Z_dot_w   [kg]
    m_added_axis ≈ rho * C_A_axis * V
    (Fossen convention: X_dot_u = -m_added_x, etc.)

  Rotational:
    K_dot_p, M_dot_q, N_dot_r   [kg*m^2]
    I_added_axis ≈ rho * C_A_rot_axis * V * L_axis^2
    (Fossen convention: K_dot_p = -I_added_roll, etc.)

--------------------------------------------------------------------
(2) QUADRATIC DRAG (translational)
  Translational quadratic damping coefficients:
    X_uu, Y_vv, Z_ww  [kg/m]  (equivalently N·s^2/m^2)
  Empirical drag:
    F_D = 0.5 * rho * C_D * A_ref * |u|u
  Fossen-style quadratic term:
    X_drag(u) = X_uu * |u|u   (with X_uu negative for opposing drag)
  Mapping:
    X_uu ≈ -0.5 * rho * C_Dx * A_x
    Y_vv ≈ -0.5 * rho * C_Dy * A_y
    Z_ww ≈ -0.5 * rho * C_Dz * A_z

  Bounding-box projected areas:
    A_x = W * H   (frontal area for surge)
    A_y = L * H   (side area for sway)
    A_z = L * W   (planform area for heave)

--------------------------------------------------------------------
(3) QUADRATIC ANGULAR DRAG (rotational)
  Rotational quadratic damping coefficients:
    K_pp, M_qq, N_rr  [kg*m^2]  (since p,q,r are in rad/s)
  Simple torque-drag scaling:
    tau_drag ≈ 0.5 * rho * C_Drot * A_rot * (L_rot^3) * |omega| omega

  Mapping to Fossen-style quadratic rotational damping:
    K_drag(p) = K_pp * |p|p    =>  K_pp ≈ -0.5 * rho * C_Dk * A_k * L_roll^3
    M_drag(q) = M_qq * |q|q    =>  M_qq ≈ -0.5 * rho * C_Dm * A_m * L_pitch^3
    N_drag(r) = N_rr * |r|r    =>  N_rr ≈ -0.5 * rho * C_Dn * A_n * L_yaw^3

  Notes:
  - This is intentionally "physics-sized" (order-of-magnitude), not geometry-exact.
  - Update A_k/A_m/A_n choices if you have CAD-derived projected areas.
"""

from dataclasses import dataclass
import math


# ============================================================
# === PARAMETERS: EDIT THESE ================================
# ============================================================

@dataclass
class Params:
    # Fluid density [kg/m^3]
    rho: float = 1000.0      # ~fresh water; use 1025.0 for seawater

    # Defender test configuration mass [kg] (used only if you choose to infer volume)
    mass_kg: float = 23.89

    # Displaced volume [m^3]
    infer_volume_from_mass: bool = True
    volume_m3: float = 0.02389  # used only if infer_volume_from_mass=False

    # ----------------------------
    # Translational C_A ranges (dimensionless)
    # ----------------------------
    CA_x_range: tuple[float, float] = (0.65, 1.91)  # surge
    CA_y_range: tuple[float, float] = (1.25, 1.34)  # sway
    CA_z_range: tuple[float, float] = (1.20, 3.50)  # heave

    # ----------------------------
    # Rotational C_A ranges (dimensionless)
    # ----------------------------
    CA_k_range: tuple[float, float] = (0.01, 0.6)    # roll
    CA_m_range: tuple[float, float] = (0.005, 0.35)  # pitch
    CA_n_range: tuple[float, float] = (0.12, 0.19)   # yaw

    # ----------------------------
    # Defender bounding-box dimensions [m]
    # ----------------------------
    L_m: float = 0.7516   # length
    W_m: float = 0.3937   # width
    H_m: float = 0.2667   # height

    # ----------------------------
    # Quadratic drag C_D ranges (dimensionless) — translational
    # ----------------------------
    CD_x_range: tuple[float, float] = (0.2, 0.9)     # surge
    CD_y_range: tuple[float, float] = (0.42, 1.49)   # sway
    CD_z_range: tuple[float, float] = (0.53, 0.91)   # heave

    # ----------------------------
    # Quadratic drag C_D ranges (dimensionless) — rotational
    # Keep broad unless you have literature back-calcs.
    # ----------------------------
    CD_k_range: tuple[float, float] = (0.05, 1.5)     # roll (about x)
    CD_m_range: tuple[float, float] = (0.3, 1.0)     # pitch (about y)
    CD_n_range: tuple[float, float] = (0.58, 1.2)     # yaw (about z)

    # ----------------------------
    # Characteristic length scales for rotational added inertia [m]
    # ----------------------------
    L_roll_m: float = 0.3937 / 2   # ~W/2
    L_pitch_m: float = 0.7516 / 2  # ~L/2
    # For yaw you may prefer sqrt((L/2)^2+(W/2)^2); keep simple unless needed
    L_yaw_m: float = 0.7516 / 2

    # Inflate sigma beyond the range-implied value (looser priors)
    sigma_inflate: float = 1.0   # e.g., 1.5 or 2.0 for looser

    # Floors to avoid overly tight priors
    sigma_floor_kg: float = 2.0           # translational added mass [kg]
    sigma_floor_kgm2: float = 0.05        # rotational added inertia [kg*m^2]
    sigma_floor_kg_per_m: float = 5.0     # translational quadratic drag [kg/m]
    sigma_floor_kgm2_quad: float = 0.05   # rotational quadratic drag [kg*m^2]

P = Params()


# ============================================================
# === CALCS ==================================================
# ============================================================

def pick_volume(p: Params) -> float:
    if p.infer_volume_from_mass:
        return p.mass_kg / p.rho
    return p.volume_m3

def _validate_pos_range(lo: float, hi: float, name: str) -> None:
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Bad {name} range: ({lo}, {hi}). Must satisfy 0 < lo < hi.")

def prior_from_CA_range_linear(rho: float, V: float, CA_lo: float, CA_hi: float,
                               sigma_inflate: float, sigma_floor: float) -> tuple[float, float]:
    _validate_pos_range(CA_lo, CA_hi, "C_A")
    CA_mean = 0.5 * (CA_lo + CA_hi)
    CA_halfspan = 0.5 * (CA_hi - CA_lo)
    mu = -rho * CA_mean * V
    sigma = rho * CA_halfspan * V
    sigma = max(sigma_floor, sigma * sigma_inflate)
    return mu, sigma

def prior_from_CA_range_rot(rho: float, V: float, L: float, CA_lo: float, CA_hi: float,
                            sigma_inflate: float, sigma_floor: float) -> tuple[float, float]:
    _validate_pos_range(CA_lo, CA_hi, "C_A(rot)")
    if L <= 0:
        raise ValueError(f"Characteristic length L must be > 0, got {L}.")
    CA_mean = 0.5 * (CA_lo + CA_hi)
    CA_halfspan = 0.5 * (CA_hi - CA_lo)
    scale = rho * V * (L ** 2)  # [kg*m^2] per unit CA
    mu = -CA_mean * scale
    sigma = CA_halfspan * scale
    sigma = max(sigma_floor, sigma * sigma_inflate)
    return mu, sigma

def prior_from_CD_range_quad_trans(rho: float, A: float, CD_lo: float, CD_hi: float,
                                   sigma_inflate: float, sigma_floor: float) -> tuple[float, float]:
    """
    Translational quadratic drag coefficient: [kg/m]
      coeff ≈ -0.5 * rho * C_D * A
    """
    _validate_pos_range(CD_lo, CD_hi, "C_D(trans)")
    if A <= 0:
        raise ValueError(f"Reference area A must be > 0, got {A}.")
    CD_mean = 0.5 * (CD_lo + CD_hi)
    CD_halfspan = 0.5 * (CD_hi - CD_lo)
    mu = -0.5 * rho * CD_mean * A
    sigma = 0.5 * rho * CD_halfspan * A
    sigma = max(sigma_floor, sigma * sigma_inflate)
    return mu, sigma

def prior_from_CD_range_quad_rot(rho: float, A: float, L: float, CD_lo: float, CD_hi: float,
                                 sigma_inflate: float, sigma_floor: float) -> tuple[float, float]:
    """
    Rotational quadratic drag coefficient: [kg*m^2]
      coeff ≈ -0.5 * rho * C_Drot * A * L^3

    where:
      A is an effective projected area for that rotation axis,
      L is a characteristic radius/length scale for that rotation axis.
    """
    _validate_pos_range(CD_lo, CD_hi, "C_D(rot)")
    if A <= 0:
        raise ValueError(f"Reference area A must be > 0, got {A}.")
    if L <= 0:
        raise ValueError(f"Characteristic length L must be > 0, got {L}.")

    CD_mean = 0.5 * (CD_lo + CD_hi)
    CD_halfspan = 0.5 * (CD_hi - CD_lo)

    scale = 0.5 * rho * A * (L ** 3)  # [kg*m^2] per unit C_D
    mu = -CD_mean * scale
    sigma = CD_halfspan * scale
    sigma = max(sigma_floor, sigma * sigma_inflate)
    return mu, sigma

def fmt(mu: float, sigma: float, units: str) -> str:
    return f"mu = {mu: .3f} {units},  sigma = {sigma: .3f} {units}"

def main():
    V = pick_volume(P)

    # ----------------------------
    # Projected areas (bounding-box)
    # ----------------------------
    A_x = P.W_m * P.H_m  # surge frontal
    A_y = P.L_m * P.H_m  # sway side
    A_z = P.L_m * P.W_m  # heave planform

    # ----------------------------
    # Effective areas for rotational drag (simple choices)
    # You can change these if you have better geometry.
    # ----------------------------
    A_k = A_y  # roll: use side area (L*H) as a rough "in-water" area
    A_m = A_x  # pitch: use frontal area (W*H)
    A_n = A_y  # yaw: use side area (L*H) (rotation in horizontal plane)

    print("\n=== Prior builder (added mass + quadratic drag + quadratic angular drag) ===")
    print(f"rho              = {P.rho:.1f} kg/m^3")
    print(f"mass_kg          = {P.mass_kg:.3f} kg")
    print(f"volume_m3        = {V:.6f} m^3  ({'inferred m/rho' if P.infer_volume_from_mass else 'user-specified'})")
    print(f"sigma_inflate    = {P.sigma_inflate:.2f}")
    print(f"sigma_floor_kg   = {P.sigma_floor_kg:.2f} kg")
    print(f"sigma_floor_kgm2 = {P.sigma_floor_kgm2:.3f} kg*m^2")
    print(f"sigma_floor_kg/m = {P.sigma_floor_kg_per_m:.2f} kg/m")
    print(f"sigma_floor_rot_quad = {P.sigma_floor_kgm2_quad:.3f} kg*m^2")

    print("\nDefender dimensions (bounding box):")
    print(f"  L = {P.L_m:.4f} m,  W = {P.W_m:.4f} m,  H = {P.H_m:.4f} m")

    print("\nProjected areas used for translational quadratic drag:")
    print(f"  A_x (surge) = W*H = {A_x:.6f} m^2")
    print(f"  A_y (sway)  = L*H = {A_y:.6f} m^2")
    print(f"  A_z (heave) = L*W = {A_z:.6f} m^2")

    print("\nCharacteristic lengths for rotational scaling:")
    print(f"  L_roll  = {P.L_roll_m:.3f} m")
    print(f"  L_pitch = {P.L_pitch_m:.3f} m")
    print(f"  L_yaw   = {P.L_yaw_m:.3f} m")

    print("\nEffective areas used for quadratic angular drag (editable heuristics):")
    print(f"  A_k (roll)  = {A_k:.6f} m^2")
    print(f"  A_m (pitch) = {A_m:.6f} m^2")
    print(f"  A_n (yaw)   = {A_n:.6f} m^2")

    # -------- Added-mass priors (translational) --------
    mu_x, sig_x = prior_from_CA_range_linear(P.rho, V, *P.CA_x_range, P.sigma_inflate, P.sigma_floor_kg)
    mu_y, sig_y = prior_from_CA_range_linear(P.rho, V, *P.CA_y_range, P.sigma_inflate, P.sigma_floor_kg)
    mu_z, sig_z = prior_from_CA_range_linear(P.rho, V, *P.CA_z_range, P.sigma_inflate, P.sigma_floor_kg)

    # -------- Added-mass priors (rotational) --------
    mu_k, sig_k = prior_from_CA_range_rot(P.rho, V, P.L_roll_m,  *P.CA_k_range, P.sigma_inflate, P.sigma_floor_kgm2)
    mu_m, sig_m = prior_from_CA_range_rot(P.rho, V, P.L_pitch_m, *P.CA_m_range, P.sigma_inflate, P.sigma_floor_kgm2)
    mu_n, sig_n = prior_from_CA_range_rot(P.rho, V, P.L_yaw_m,   *P.CA_n_range, P.sigma_inflate, P.sigma_floor_kgm2)

    # -------- Quadratic drag priors (translational) --------
    mu_xuu, sig_xuu = prior_from_CD_range_quad_trans(P.rho, A_x, *P.CD_x_range, P.sigma_inflate, P.sigma_floor_kg_per_m)
    mu_yvv, sig_yvv = prior_from_CD_range_quad_trans(P.rho, A_y, *P.CD_y_range, P.sigma_inflate, P.sigma_floor_kg_per_m)
    mu_zww, sig_zww = prior_from_CD_range_quad_trans(P.rho, A_z, *P.CD_z_range, P.sigma_inflate, P.sigma_floor_kg_per_m)

    # -------- Quadratic angular drag priors (rotational) --------
    mu_kpp, sig_kpp = prior_from_CD_range_quad_rot(P.rho, A_k, P.L_roll_m,  *P.CD_k_range, P.sigma_inflate, P.sigma_floor_kgm2_quad)
    mu_mqq, sig_mqq = prior_from_CD_range_quad_rot(P.rho, A_m, P.L_pitch_m, *P.CD_m_range, P.sigma_inflate, P.sigma_floor_kgm2_quad)
    mu_nrr, sig_nrr = prior_from_CD_range_quad_rot(P.rho, A_n, P.L_yaw_m,   *P.CD_n_range, P.sigma_inflate, P.sigma_floor_kgm2_quad)

    print("\n--- Priors (Normal, weakly-informative) ---")

    print("\nAdded mass (Fossen coefficients; expected NEGATIVE):")
    print(f"X_dot_u  (surge) from C_Ax in {P.CA_x_range}: {fmt(mu_x, sig_x, 'kg')}")
    print(f"Y_dot_v  (sway)  from C_Ay in {P.CA_y_range}: {fmt(mu_y, sig_y, 'kg')}")
    print(f"Z_dot_w  (heave) from C_Az in {P.CA_z_range}: {fmt(mu_z, sig_z, 'kg')}")
    print(f"K_dot_p  (roll)  from C_Ak in {P.CA_k_range}: {fmt(mu_k, sig_k, 'kg*m^2')}")
    print(f"M_dot_q  (pitch) from C_Am in {P.CA_m_range}: {fmt(mu_m, sig_m, 'kg*m^2')}")
    print(f"N_dot_r  (yaw)   from C_An in {P.CA_n_range}: {fmt(mu_n, sig_n, 'kg*m^2')}")

    print("\nQuadratic drag (translational; expected NEGATIVE):")
    print(f"X_uu  (surge) from C_Dx in {P.CD_x_range}: {fmt(mu_xuu, sig_xuu, 'kg/m')}")
    print(f"Y_vv  (sway)  from C_Dy in {P.CD_y_range}: {fmt(mu_yvv, sig_yvv, 'kg/m')}")
    print(f"Z_ww  (heave) from C_Dz in {P.CD_z_range}: {fmt(mu_zww, sig_zww, 'kg/m')}")

    print("\nQuadratic angular drag (rotational; expected NEGATIVE):")
    print(f"K_pp  (roll)  from C_Dk in {P.CD_k_range}: {fmt(mu_kpp, sig_kpp, 'kg*m^2')}")
    print(f"M_qq  (pitch) from C_Dm in {P.CD_m_range}: {fmt(mu_mqq, sig_mqq, 'kg*m^2')}")
    print(f"N_rr  (yaw)   from C_Dn in {P.CD_n_range}: {fmt(mu_nrr, sig_nrr, 'kg*m^2')}")

    print("\nNotes:")
    print("- Added-mass and quadratic-drag coefficients are expected to be NEGATIVE in the Fossen convention.")
    print("- If you enforce sign constraints, truncate each coefficient to (-inf, 0].")
    print("- Translational quadratic drag uses bounding-box projected areas; replace with CAD projected areas if available.")
    print("- Rotational quadratic drag uses a simple torque-drag scaling ~ 0.5*rho*C_D*A*L^3; treat these as order-of-magnitude priors.")
    print("- Update C_Dk/C_Dm/C_Dn ranges if you back-calc angular drag from prior studies or from your own MLE consistency checks.")
    print("- To loosen further: increase sigma_inflate (e.g., 1.5–3.0) and/or widen the C_A / C_D ranges.")

if __name__ == "__main__":
    main()
