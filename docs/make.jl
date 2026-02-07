using AtmosphericDynamics
using Documenter

DocMeta.setdocmeta!(
    AtmosphericDynamics, :DocTestSetup,
    :(using AtmosphericDynamics); recursive = true
)

makedocs(;
    modules = [AtmosphericDynamics],
    authors = "EarthSciML authors and contributors",
    repo = "https://github.com/EarthSciML/AtmosphericDynamics.jl/blob/{commit}{path}#{line}",
    sitename = "AtmosphericDynamics.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://EarthSciML.github.io/AtmosphericDynamics.jl",
        edit_link = "main",
        assets = String[],
        repolink = "https://github.com/EarthSciML/AtmosphericDynamics.jl"
    ),
    pages = [
        "Home" => "index.md",
        "Atmospheric Fundamentals" => "seinfeld_pandis_ch1.md",
        "General Circulation" => "general_circulation.md",
        "Local Scale Meteorology" => "local_scale_meteorology.md",
        "Atmospheric Diffusion" => "atmospheric_diffusion.md",
        "Global Cycles" => "global_cycles.md",
        "Holtslag & Boville 1993" => "holtslag_boville_1993.md",
    ]
)

deploydocs(;
    repo = "github.com/EarthSciML/AtmosphericDynamics.jl",
    devbranch = "main"
)
