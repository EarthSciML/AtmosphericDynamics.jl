# AtmosphericDynamics.jl

Models of atmospheric dynamics implemented in [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), part of the [EarthSciML](https://github.com/EarthSciML) ecosystem.

## Components

- **[Atmospheric Fundamentals](@ref)**: Ideal gas law, scale height, barometric formula, mixing ratios, saturation vapor pressure, and relative humidity (Seinfeld & Pandis Ch. 1)
- **[General Circulation of the Atmosphere](@ref)**: Coriolis parameter, geostrophic wind, and thermal wind relations (Seinfeld & Pandis Ch. 21)
- **[Local Scale Meteorology](@ref)**: Atmospheric stability, Monin-Obukhov similarity theory, and Pasquill stability classes (Seinfeld & Pandis Ch. 16)
- **[Atmospheric Diffusion Parameterizations](@ref)**: Wind profiles and eddy diffusivities for unstable, neutral, and stable conditions (Seinfeld & Pandis Ch. 18)
- **[Global Cycles: Sulfur and Carbon](@ref)**: Sulfur cycle, carbon cycle, and four-compartment atmosphere models (Seinfeld & Pandis Ch. 22)
- **[Holtslag & Boville (1993) Boundary Layer Diffusion](@ref)**: Surface fluxes, local diffusion, and nonlocal boundary layer schemes
