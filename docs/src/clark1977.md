# Clark (1977): Mountain Wave Model

## Overview

This module implements the key equations from Clark's seminal paper on small-scale
atmospheric dynamics with terrain-following coordinates. The model describes
nonhydrostatic, anelastic flow over irregular terrain and was one of the first
to demonstrate the terrain-following coordinate transformation for mountain wave
simulations.

The implementation includes:
- **Isentropic base-state** thermodynamic profiles (Eqs. 2.6-2.9)
- **"Witch of Agnesi"** bell-shaped mountain topography (Eq. 7.1)
- **Terrain-following coordinate transformation** (Eqs. 2.20-2.23)
- **Smagorinsky turbulence closure** (Eqs. 2.15-2.19)
- **Linearized 2D mountain wave equations** as a PDESystem

**Reference**: Clark, T. L. (1977). "A Small-Scale Dynamic Model Using a
Terrain-Following Coordinate Transformation." *Journal of Computational Physics*,
24, 186-215.

```@docs
IsentropicBaseState
WitchOfAgnesi
TerrainFollowingTransform
SmagorinskyTurbulence
MountainWave2D
```

## Implementation

### Isentropic Base State (Eqs. 2.6-2.9)

The base state assumes constant potential temperature ``\bar{\theta}(z) = \Theta``
and derives pressure, temperature, and density profiles in hydrostatic balance.

```@example clark1977
using DataFrames, ModelingToolkit, ModelingToolkit: mtkcompile, Symbolics, @named
using DynamicQuantities
using AtmosphericDynamics

@named bs = IsentropicBaseState()
sys = mtkcompile(bs)
vars = unknowns(sys)
DataFrame(
    :Name => [string(Symbolics.tosymbol(v, escape=false)) for v in vars],
    :Units => [ModelingToolkit.get_unit(v) for v in vars],
    :Description => [ModelingToolkit.getdescription(v) for v in vars]
)
```

### Witch of Agnesi Topography (Eq. 7.1)

```@example clark1977
@named topo = WitchOfAgnesi()
sys_topo = mtkcompile(topo)
vars_topo = unknowns(sys_topo)
DataFrame(
    :Name => [string(Symbolics.tosymbol(v, escape=false)) for v in vars_topo],
    :Units => [ModelingToolkit.get_unit(v) for v in vars_topo],
    :Description => [ModelingToolkit.getdescription(v) for v in vars_topo]
)
```

### Terrain-Following Coordinate Transform (Eqs. 2.20-2.23)

```@example clark1977
@named tft = TerrainFollowingTransform()
sys_tft = mtkcompile(tft)
vars_tft = unknowns(sys_tft)
DataFrame(
    :Name => [string(Symbolics.tosymbol(v, escape=false)) for v in vars_tft],
    :Units => [ModelingToolkit.get_unit(v) for v in vars_tft],
    :Description => [ModelingToolkit.getdescription(v) for v in vars_tft]
)
```

### Smagorinsky Turbulence Closure (Eqs. 2.15-2.19)

```@example clark1977
@named smag = SmagorinskyTurbulence()
sys_smag = mtkcompile(smag)
vars_smag = unknowns(sys_smag)
DataFrame(
    :Name => [string(Symbolics.tosymbol(v, escape=false)) for v in vars_smag],
    :Units => [ModelingToolkit.get_unit(v) for v in vars_smag],
    :Description => [ModelingToolkit.getdescription(v) for v in vars_smag]
)
```

### Mountain Wave PDESystem

```@example clark1977
pde = MountainWave2D()
println("Equations: ", length(pde.eqs))
println("Boundary/Initial conditions: ", length(pde.bcs))
println("Independent variables: ", length(pde.ivs))
println("Dependent variables: ", length(pde.dvs))
```

## Analysis

### Base-State Atmosphere Profiles

The isentropic base state produces temperature, pressure, and density profiles
that decrease with height, consistent with hydrostatic balance (Eq. 2.12).

```@example clark1977
using NonlinearSolve
using CairoMakie

@named bs = IsentropicBaseState()
sys = mtkcompile(bs)

z_vals = 0.0:100.0:10000.0
T_vals = Float64[]
p_vals = Float64[]
ρ_vals = Float64[]

for z in z_vals
    sol = solve(NonlinearProblem(sys, Dict(sys.z => z)), NewtonRaphson())
    push!(T_vals, sol[sys.T_bar])
    push!(p_vals, sol[sys.p_bar])
    push!(ρ_vals, sol[sys.ρ_bar])
end

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Height (m)",
    title="Base-State Temperature")
lines!(ax1, T_vals, collect(z_vals))

ax2 = Axis(fig[1, 2], xlabel="Pressure (Pa)", ylabel="Height (m)",
    title="Base-State Pressure")
lines!(ax2, p_vals, collect(z_vals))

ax3 = Axis(fig[1, 3], xlabel="Density (kg/m³)", ylabel="Height (m)",
    title="Base-State Density")
lines!(ax3, ρ_vals, collect(z_vals))

fig
```

### Witch of Agnesi Mountain Profile

The "Witch of Agnesi" topography ``z_s = a^2 h / (a^2 + x^2)`` is shown
for both the 100 m and 1 km mountain heights used in the paper.

```@example clark1977
@named topo = WitchOfAgnesi()
sys_topo = mtkcompile(topo)

x_vals = -15000.0:100.0:15000.0

fig2 = Figure(size=(700, 400))
ax = Axis(fig2[1, 1], xlabel="x (m)", ylabel="z_s (m)",
    title="Witch of Agnesi Mountain Profile (Eq. 7.1)")

for (h, label) in [(100.0, "h = 100 m"), (1000.0, "h = 1 km")]
    zs = Float64[]
    for x in x_vals
        sol = solve(NonlinearProblem(sys_topo, Dict(
            sys_topo.x => x, sys_topo.h_mtn => h)), NewtonRaphson())
        push!(zs, sol[sys_topo.z_s])
    end
    lines!(ax, collect(x_vals), zs, label=label)
end
axislegend(ax)

fig2
```

### Terrain-Following Coordinate Grid

The terrain-following transformation maps the irregular lower boundary to a
flat computational surface. Here we visualize the physical grid lines for the
1 km mountain case.

```@example clark1977
@named tft = TerrainFollowingTransform()
sys_tft = mtkcompile(tft)
@named topo2 = WitchOfAgnesi()
sys_topo2 = mtkcompile(topo2)

H_val = 8000.0
h_val = 1000.0
a_val = 3000.0
x_range = -12000.0:600.0:12000.0
zbar_range = 0.0:400.0:H_val

fig3 = Figure(size=(800, 400))
ax3 = Axis(fig3[1, 1], xlabel="x (m)", ylabel="z (m)",
    title="Terrain-Following Grid (h = 1 km)")

for zbar in zbar_range
    z_line = Float64[]
    for x in x_range
        topo_sol = solve(NonlinearProblem(sys_topo2, Dict(
            sys_topo2.x => x, sys_topo2.h_mtn => h_val, sys_topo2.a => a_val)),
            NewtonRaphson())
        zs = topo_sol[sys_topo2.z_s]

        tft_sol = solve(NonlinearProblem(sys_tft, Dict(
            sys_tft.z_s => zs, sys_tft.z_bar => zbar, sys_tft.H => H_val)),
            NewtonRaphson())
        push!(z_line, tft_sol[sys_tft.z_phys])
    end
    lines!(ax3, collect(x_range), z_line, color=:gray, linewidth=0.5)
end

# Draw terrain surface
zs_surf = Float64[]
for x in x_range
    sol = solve(NonlinearProblem(sys_topo2, Dict(
        sys_topo2.x => x, sys_topo2.h_mtn => h_val, sys_topo2.a => a_val)),
        NewtonRaphson())
    push!(zs_surf, sol[sys_topo2.z_s])
end
band!(ax3, collect(x_range), zeros(length(x_range)), zs_surf,
    color=(:brown, 0.3))
lines!(ax3, collect(x_range), zs_surf, color=:brown, linewidth=2)

fig3
```

### Smagorinsky Eddy Viscosity

The Smagorinsky closure relates eddy viscosity to the resolved deformation
rate. This plot shows ``K_M`` as a function of the dominant shear ``D_{13}``
for ``\Delta = 600`` m and ``k = 0.25``.

```@example clark1977
@named smag = SmagorinskyTurbulence()
sys_smag = mtkcompile(smag)

D13_vals = 0.0:0.001:0.02
K_M_vals = Float64[]
for d13 in D13_vals
    sol = solve(NonlinearProblem(sys_smag, Dict(
        sys_smag.D13 => d13, sys_smag.Δ_grid => 600.0)), NewtonRaphson())
    push!(K_M_vals, sol[sys_smag.K_M])
end

fig4 = Figure(size=(600, 400))
ax4 = Axis(fig4[1, 1], xlabel="D₁₃ (1/s)", ylabel="K_M (m²/s)",
    title="Smagorinsky Eddy Viscosity (Eq. 2.17, k=0.25, Δ=600m)")
lines!(ax4, collect(D13_vals), K_M_vals)

fig4
```
