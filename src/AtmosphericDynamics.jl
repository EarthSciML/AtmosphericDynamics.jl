module AtmosphericDynamics

using DocStringExtensions
using ModelingToolkit: t, D, System, @variables, @parameters, @named,
    @constants, @component, Differential, PDESystem
using DynamicQuantities: @u_str
using DomainSets: Interval
using EarthSciMLBase

include("seinfeld_pandis_ch1.jl")
include("general_circulation.jl")
include("local_scale_meteorology.jl")
include("atmospheric_diffusion.jl")
include("global_cycles.jl")
include("holtslag_boville_1993.jl")
include("clark1977.jl")

end # module
