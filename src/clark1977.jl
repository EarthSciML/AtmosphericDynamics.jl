"""
    Clark (1977): A Small-Scale Dynamic Model Using a Terrain-Following
    Coordinate Transformation

ModelingToolkit.jl implementation of equations from:
Clark, T. L. (1977). "A Small-Scale Dynamic Model Using a Terrain-Following
Coordinate Transformation." Journal of Computational Physics, 24, 186-215.

This module implements:
- Equations 2.6-2.9: Isentropic base-state thermodynamic profiles
- Equation 7.1: "Witch of Agnesi" mountain topography
- Equations 2.20-2.23, 2.27: Terrain-following coordinate transformation
- Equations 2.15-2.19: Smagorinsky turbulence closure
- Linearized 2D Boussinesq mountain wave PDESystem
"""

export IsentropicBaseState, WitchOfAgnesi, TerrainFollowingTransform
export SmagorinskyTurbulence, MountainWave2D

#=============================================================================
# Equations 2.6-2.9: Isentropic Base-State Atmosphere
=============================================================================#

"""
$(TYPEDSIGNATURES)

Isentropic base-state atmosphere profiles from the deep equations of
Ogura and Phillips (1962), as used in Clark (1977).

The base state assumes constant potential temperature ``\\bar{\\theta}(z) = \\Theta``
(Eq. 2.6) and derives temperature, pressure, and density profiles that satisfy
hydrostatic balance (Eq. 2.12).

**Reference**: Clark (1977), Eqs. 2.6-2.9.

Parameters:
- `Θ`: Reference potential temperature (K)
- `P_0`: Surface pressure (Pa)
- `z`: Height above surface (m)

Variables (outputs):
- `T_bar`: Base-state temperature ``\\bar{T}(z) = \\Theta(1 - z/H_s)``
- `p_bar`: Base-state pressure ``\\bar{p}(z) = P_0(1 - z/H_s)^{1/\\kappa}``
- `ρ_bar`: Base-state density ``\\bar{\\rho}(z)``
- `H_s`: Isentropic scale height ``H_s = C_p \\Theta / g``
"""
@component function IsentropicBaseState(; name = :IsentropicBaseState)
    @constants begin
        g = 9.81, [description = "Gravitational acceleration", unit = u"m/s^2"]
        R_d = 287.0, [description = "Dry air gas constant", unit = u"J/(kg*K)"]
        c_p = 1004.0, [
                description = "Specific heat at constant pressure",
                unit = u"J/(kg*K)",
            ]
        inv_kappa = c_p / R_d,
            [description = "C_p/R_d = 1/κ, pressure profile exponent (dimensionless)", unit = u"1"]
        inv_kappa_m1 = c_p / R_d - 1.0,
            [description = "C_p/R_d - 1, density profile exponent (dimensionless)", unit = u"1"]
    end

    @parameters begin
        Θ = 300.0, [description = "Reference potential temperature", unit = u"K"]
        P_0 = 1.0e5, [description = "Surface pressure", unit = u"Pa"]
        z = 0.0, [description = "Height above surface", unit = u"m"]
    end

    @variables begin
        T_bar(t), [description = "Base-state temperature (Eq. 2.7)", unit = u"K"]
        p_bar(t), [description = "Base-state pressure (Eq. 2.8)", unit = u"Pa"]
        ρ_bar(t), [description = "Base-state density (Eq. 2.9)", unit = u"kg/m^3"]
        H_s(t), [description = "Isentropic scale height", unit = u"m"]
    end

    eqs = [
        H_s ~ c_p * Θ / g,                                          # Scale height
        T_bar ~ Θ * (1 - z / H_s),                                  # Eq. 2.7
        p_bar ~ P_0 * (1 - z / H_s)^inv_kappa,                      # Eq. 2.8
        ρ_bar ~ P_0 / (R_d * Θ) * (1 - z / H_s)^inv_kappa_m1,      # Eq. 2.9
    ]

    return System(eqs, t; name)
end

#=============================================================================
# Equation 7.1: Witch of Agnesi Mountain Topography
=============================================================================#

"""
$(TYPEDSIGNATURES)

"Witch of Agnesi" bell-shaped mountain topography used as a test case
for mountain wave simulations.

Implements Equation 7.1:
``z_s = a^2 h / (a^2 + x^2)``

where:
- `a`: Mountain half-width (m)
- `h_mtn`: Mountain peak height (m)
- `x`: Horizontal position relative to mountain center (m)
- `z_s`: Terrain surface height (m)

**Reference**: Clark (1977), Eq. 7.1.
"""
@component function WitchOfAgnesi(; name = :WitchOfAgnesi)
    @parameters begin
        a = 3000.0, [description = "Mountain half-width", unit = u"m"]
        h_mtn = 100.0, [description = "Mountain peak height", unit = u"m"]
        x = 0.0, [description = "Horizontal position", unit = u"m"]
    end

    @variables begin
        z_s(t), [description = "Terrain surface height (Eq. 7.1)", unit = u"m"]
        dz_s_dx(t),
            [description = "Terrain slope ∂z_s/∂x (dimensionless)", unit = u"1"]
    end

    eqs = [
        z_s ~ a^2 * h_mtn / (a^2 + x^2),                     # Eq. 7.1
        dz_s_dx ~ -2 * a^2 * h_mtn * x / (a^2 + x^2)^2,      # ∂/∂x of Eq. 7.1
    ]

    return System(eqs, t; name)
end

#=============================================================================
# Equations 2.20-2.23, 2.27: Terrain-Following Coordinate Transformation
=============================================================================#

"""
$(TYPEDSIGNATURES)

Terrain-following coordinate transformation from physical ``(x, y, z)`` to
computational ``(x, y, \\bar{z})`` coordinates.

The transformation ``\\bar{z} = H(z - z_s)/(H - z_s)`` (Eq. 2.20) maps the
irregular lower boundary ``z = z_s(x,y)`` to the flat surface ``\\bar{z} = 0``
and preserves the upper boundary ``z = H`` at ``\\bar{z} = H``.

**Reference**: Clark (1977), Eqs. 2.20-2.23, 2.27.

Parameters:
- `z_s`: Terrain surface height (m)
- `H`: Domain height (m)
- `z_bar`: Transformed vertical coordinate (m)
- `dz_s_dx`: Terrain slope in x direction (dimensionless)
- `dz_s_dy`: Terrain slope in y direction (dimensionless)

Variables (outputs):
- `G_sqrt`: Square root of the Jacobian determinant ``G^{1/2} = 1 - z_s/H``
- `G_sqrt_G13`: Metric tensor product ``G^{1/2} G^{13}``
- `G_sqrt_G23`: Metric tensor product ``G^{1/2} G^{23}``
- `z_phys`: Physical height corresponding to ``\\bar{z}``
"""
@component function TerrainFollowingTransform(; name = :TerrainFollowingTransform)
    @parameters begin
        z_s = 0.0, [description = "Terrain surface height", unit = u"m"]
        H = 8000.0, [description = "Domain height", unit = u"m"]
        z_bar = 0.0, [description = "Transformed vertical coordinate", unit = u"m"]
        dz_s_dx = 0.0,
            [description = "Terrain slope ∂z_s/∂x (dimensionless)", unit = u"1"]
        dz_s_dy = 0.0,
            [description = "Terrain slope ∂z_s/∂y (dimensionless)", unit = u"1"]
    end

    @variables begin
        G_sqrt(t),
            [description = "Jacobian G^(1/2) = 1 - z_s/H (Eq. 2.21) (dimensionless)", unit = u"1"]
        G_sqrt_G13(t),
            [description = "Metric coefficient G^(1/2)·G^13 (Eq. 2.22) (dimensionless)", unit = u"1"]
        G_sqrt_G23(t),
            [description = "Metric coefficient G^(1/2)·G^23 (Eq. 2.23) (dimensionless)", unit = u"1"]
        z_phys(t), [description = "Physical height z (inverse of Eq. 2.20)", unit = u"m"]
        ω_diag(t),
            [description = "Diagnostic factor for ω: ω·G^(1/2) = w + G^(1/2)G^13·u + G^(1/2)G^23·v (Eq. 2.27) (dimensionless)", unit = u"1"]
    end

    eqs = [
        G_sqrt ~ 1 - z_s / H,                                        # Eq. 2.21
        G_sqrt_G13 ~ (z_bar / H - 1) * dz_s_dx,                      # Eq. 2.22
        G_sqrt_G23 ~ (z_bar / H - 1) * dz_s_dy,                      # Eq. 2.23
        z_phys ~ z_bar * (H - z_s) / H + z_s,                        # Inverse of Eq. 2.20
        ω_diag ~ G_sqrt,                                              # Factor for Eq. 2.27: ω = (w + G^(1/2)G^13·u + G^(1/2)G^23·v) / G^(1/2)
    ]

    return System(eqs, t; name)
end

#=============================================================================
# Equations 2.15-2.19: Smagorinsky Turbulence Closure
=============================================================================#

"""
$(TYPEDSIGNATURES)

Smagorinsky (1963) subgrid-scale turbulence closure for eddy viscosity
and turbulent heat diffusivity.

Implements the eddy viscosity coefficient (Eq. 2.17):
``K_M = (k \\Delta)^2 |\\text{Def}|``

where the total deformation is (Eq. 2.18):
``\\text{Def}^2 = \\frac{1}{2}(D_{11}^2 + D_{22}^2 + D_{33}^2) + D_{12}^2 + D_{13}^2 + D_{23}^2``

and ``k = 0.25`` is the Smagorinsky constant.

The turbulent heat diffusivity is assumed equal to the eddy viscosity:
``K_H = K_M`` (Eq. 2.19).

**Reference**: Clark (1977), Eqs. 2.15-2.19.
"""
@component function SmagorinskyTurbulence(; name = :SmagorinskyTurbulence)
    @constants begin
        k_smag = 0.25,
            [description = "Smagorinsky constant (Eq. 2.17) (dimensionless)", unit = u"1"]
        Def_ref = 1.0,
            [
                description = "Reference deformation rate for non-dimensionalization",
                unit = u"1/s",
            ]
        two_thirds = 2.0/3.0,
            [description = "Coefficient 2/3 in deformation tensor trace subtraction (dimensionless)", unit = u"1"]
    end

    @parameters begin
        Δ_grid = 600.0, [description = "Grid resolution", unit = u"m"]
        D11 = 0.0, [description = "Deformation tensor D₁₁ (Eq. 2.16)", unit = u"1/s"]
        D22 = 0.0, [description = "Deformation tensor D₂₂ (Eq. 2.16)", unit = u"1/s"]
        D33 = 0.0, [description = "Deformation tensor D₃₃ (Eq. 2.16)", unit = u"1/s"]
        D12 = 0.0, [description = "Deformation tensor D₁₂ (Eq. 2.16)", unit = u"1/s"]
        D13 = 0.0, [description = "Deformation tensor D₁₃ (Eq. 2.16)", unit = u"1/s"]
        D23 = 0.0, [description = "Deformation tensor D₂₃ (Eq. 2.16)", unit = u"1/s"]
    end

    @variables begin
        Def(t),
            [description = "Total deformation |Def| (Eq. 2.18)", unit = u"1/s"]
        K_M(t),
            [description = "Eddy viscosity coefficient (Eq. 2.17)", unit = u"m^2/s"]
        K_H(t),
            [description = "Eddy heat diffusivity (Eq. 2.19, K_H = K_M)", unit = u"m^2/s"]
    end

    # Implement the full deformation tensor from Eq. 2.16
    # D_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) - (2/3)δ_ij ∂u_k/∂x_k
    # The input D_ij values should already include the trace subtraction
    # Non-dimensionalize before sqrt to handle units correctly
    # Def² has units 1/s², sqrt(1/s²) = 1/s
    Def_sq_dimless = (
        0.5 * (D11^2 + D22^2 + D33^2) +
            D12^2 + D13^2 + D23^2
    ) / Def_ref^2

    eqs = [
        Def ~ Def_ref * sqrt(Def_sq_dimless),                         # Eq. 2.18
        K_M ~ (k_smag * Δ_grid)^2 * Def,                             # Eq. 2.17
        K_H ~ K_M,                                                    # Eq. 2.19
    ]

    return System(eqs, t; name)
end

#=============================================================================
# Linearized 2D Boussinesq Mountain Wave PDESystem
=============================================================================#

"""
$(TYPEDSIGNATURES)

Construct a `PDESystem` for 2D linearized Boussinesq mountain wave flow
over terrain-following coordinates.

Implements a simplified form of the Clark (1977) equations for demonstration
purposes. This uses linearized forms of the governing equations for flow over
a "Witch of Agnesi" mountain ridge with pseudo-compressible formulation.

The governing equations are linearized forms of Clark's Eqs. 2.1, 2.3, 2.4, 2.14:

- **u-momentum**: ``∂u'/∂t = -U_0 ∂u'/∂x - (1/ρ_0) ∂p'/∂x + ν ∇²u'``
- **w-momentum**: ``∂w'/∂t = -U_0 ∂w'/∂x - (1/ρ_0) ∂p'/∂z + (g/Θ) θ' + ν ∇²w'``
- **Thermodynamics**: ``∂θ'/∂t = -U_0 ∂θ'/∂x - (N²Θ/g) w' + κ ∇²θ'``
- **Pseudo-compressible continuity**: ``∂p'/∂t = -C_a² ρ_0 (∂u'/∂x + ∂w'/∂z)``

The lower boundary kinematic condition (from Eq. 7.1) is:
``w'(x, 0) = U_0 ∂z_s/∂x`` where ``z_s = a²h/(a² + x²)``

**NOTE**: This is a simplified demonstration. The full Clark (1977) model
includes nonlinear terms, full terrain-following coordinate transformation,
and more sophisticated boundary conditions detailed in Sections 3-4.

**Reference**: Clark (1977), Eqs. 2.1, 2.3, 2.4, 2.14, 7.1; Table I.

# Parameters (from Clark 1977, Table I and Section 7)
- `U_0_val=4.0`: Mean flow velocity (m/s) [Eq. 7.3, Table I]
- `N_val=0.01`: Brunt-Väisälä frequency (1/s) [from dθ/dz = 3K/km, Eq. 7.2]
- `Θ_val=300.0`: Reference potential temperature (K) [typical atmospheric value]
- `ρ_0_val=1.225`: Reference density (kg/m³) [standard atmosphere]
- `C_a_val=50.0`: Pseudo-compressible acoustic speed (m/s) [numerical parameter]
- `ν_val=20.0`: Eddy viscosity (m²/s) [from Smagorinsky closure]
- `κ_val=20.0`: Eddy thermal diffusivity (m²/s) [K_H = K_M assumption]
- `a_val=3000.0`: Mountain half-width (m) [Table I, 3 km]
- `h_val=100.0`: Mountain height (m) [Table I, cases 14, 18]
- `L_val=18000.0`: Half domain width (m) [≈6a for minimal boundary effects]
- `H_val=8000.0`: Domain height (m) [typical atmospheric model depth]
- `T_end_val=4000.0`: Simulation end time (s) [≈1 hr mountain wave response]
"""
function MountainWave2D(;
        U_0_val = 4.0,
        N_val = 0.01,
        Θ_val = 300.0,
        ρ_0_val = 1.225,
        C_a_val = 50.0,
        ν_val = 20.0,
        κ_val = 20.0,
        a_val = 3000.0,
        h_val = 100.0,
        L_val = 18000.0,
        H_val = 8000.0,
        T_end_val = 4000.0,
        name = :MountainWave2D,
    )
    # Proper unit tracking is maintained throughout. MethodOfLines discretization
    # uses checks=false only for the discretization step, as per project standards.
    # All values use proper SI units with DynamicQuantities annotations.

    # Independent variables with proper units
    @parameters t_pde [unit = u"s"]
    @parameters x_coord [unit = u"m"]
    @parameters z_coord [unit = u"m"]

    Dt = Differential(t_pde)
    Dx = Differential(x_coord)
    Dz = Differential(z_coord)
    Dxx = Dx ∘ Dx
    Dzz = Dz ∘ Dz

    # Physical parameters with proper unit annotations
    @parameters begin
        U_0 = U_0_val, [description = "Mean flow velocity (Eq. 7.3)", unit = u"m/s"]
        N_bv = N_val, [description = "Brunt-Väisälä frequency", unit = u"1/s"]
        Θ_ref = Θ_val, [description = "Reference potential temperature", unit = u"K"]
        ρ_0 = ρ_0_val, [description = "Reference density", unit = u"kg/m^3"]
        C_a = C_a_val, [description = "Pseudo-compressible acoustic speed", unit = u"m/s"]
        ν = ν_val, [description = "Eddy viscosity", unit = u"m^2/s"]
        κ_diff = κ_val, [description = "Eddy thermal diffusivity", unit = u"m^2/s"]
        a_mtn = a_val, [description = "Mountain half-width (Eq. 7.1)", unit = u"m"]
        h_mtn = h_val, [description = "Mountain height (Eq. 7.1)", unit = u"m"]
    end

    # Dependent variables with proper unit annotations
    @variables begin
        u_p(..), [description = "x-velocity perturbation", unit = u"m/s"]
        w_p(..), [description = "z-velocity perturbation", unit = u"m/s"]
        θ_p(..), [description = "Potential temperature perturbation", unit = u"K"]
        p_p(..), [description = "Pressure perturbation", unit = u"Pa"]
    end

    # Constants with proper units
    @constants begin
        g = 9.81, [description = "Gravitational acceleration", unit = u"m/s^2"]
    end

    # Convenience aliases
    u = u_p(t_pde, x_coord, z_coord)
    w = w_p(t_pde, x_coord, z_coord)
    θ = θ_p(t_pde, x_coord, z_coord)
    p = p_p(t_pde, x_coord, z_coord)

    # Governing equations (linearized 2D Boussinesq with pseudo-compressible pressure)
    eqs = [
        # u-momentum equation (linearized form of Eq. 2.1, no Coriolis)
        Dt(u) ~ -U_0 * Dx(u) - Dx(p) / ρ_0 +
            ν * (Dxx(u) + Dzz(u)),

        # w-momentum equation (linearized form of Eq. 2.3, with buoyancy)
        Dt(w) ~ -U_0 * Dx(w) - Dz(p) / ρ_0 + g * θ / Θ_ref +
            ν * (Dxx(w) + Dzz(w)),

        # Potential temperature equation (linearized form of Eq. 2.14)
        # Source term: -w * N²Θ/g represents advection against the stable stratification
        Dt(θ) ~ -U_0 * Dx(θ) - N_bv^2 * Θ_ref / g * w +
            κ_diff * (Dxx(θ) + Dzz(θ)),

        # Pseudo-compressible mass continuity (adapted from Eq. 2.4)
        # In the anelastic limit (C_a → ∞), this enforces ∂u/∂x + ∂w/∂z → 0
        Dt(p) ~ -C_a^2 * ρ_0 * (Dx(u) + Dz(w)),
    ]

    # Mountain forcing: w'(x, 0) = U₀ · ∂z_s/∂x where z_s = a²h/(a² + x²) (Eq. 7.1)
    # ∂z_s/∂x = -2a²hx / (a² + x²)²
    w_mountain = -2 * U_0 * a_mtn^2 * h_mtn * x_coord /
        (a_mtn^2 + x_coord^2)^2

    # Boundary and initial conditions
    bcs = [
        # Initial conditions (all zero — undisturbed atmosphere)
        u_p(0, x_coord, z_coord) ~ 0.0,
        w_p(0, x_coord, z_coord) ~ 0.0,
        θ_p(0, x_coord, z_coord) ~ 0.0,
        p_p(0, x_coord, z_coord) ~ 0.0,

        # Lower boundary z = 0
        w_p(t_pde, x_coord, 0) ~ w_mountain,                 # Linearized kinematic BC (Eq. 7.1)
        Dz(u_p(t_pde, x_coord, 0)) ~ 0.0,                    # Free slip (Eq. 3.32)
        Dz(θ_p(t_pde, x_coord, 0)) ~ 0.0,                    # Insulating surface (Eq. 3.43)
        Dz(p_p(t_pde, x_coord, 0)) ~ 0.0,                    # Pressure BC

        # Upper boundary z = H
        w_p(t_pde, x_coord, H_val) ~ 0.0,                    # Rigid lid (Eq. 3.32)
        Dz(u_p(t_pde, x_coord, H_val)) ~ 0.0,                # Free slip (Eq. 3.32)
        Dz(θ_p(t_pde, x_coord, H_val)) ~ 0.0,                # Insulating (Eq. 3.42)
        Dz(p_p(t_pde, x_coord, H_val)) ~ 0.0,                # Pressure BC

        # Left boundary x = -L (undisturbed upstream)
        u_p(t_pde, -L_val, z_coord) ~ 0.0,
        w_p(t_pde, -L_val, z_coord) ~ 0.0,
        θ_p(t_pde, -L_val, z_coord) ~ 0.0,
        p_p(t_pde, -L_val, z_coord) ~ 0.0,

        # Right boundary x = L (undisturbed downstream)
        u_p(t_pde, L_val, z_coord) ~ 0.0,
        w_p(t_pde, L_val, z_coord) ~ 0.0,
        θ_p(t_pde, L_val, z_coord) ~ 0.0,
        p_p(t_pde, L_val, z_coord) ~ 0.0,
    ]

    # Domain specification
    domains = [
        t_pde ∈ Interval(0.0, T_end_val),
        x_coord ∈ Interval(-L_val, L_val),
        z_coord ∈ Interval(0.0, H_val),
    ]

    return PDESystem(
        eqs, bcs, domains,
        [t_pde, x_coord, z_coord],
        [
            u_p(t_pde, x_coord, z_coord), w_p(t_pde, x_coord, z_coord),
            θ_p(t_pde, x_coord, z_coord), p_p(t_pde, x_coord, z_coord),
        ],
        [U_0, N_bv, Θ_ref, ρ_0, C_a, ν, κ_diff, a_mtn, h_mtn, g];
        name,
    )
end
