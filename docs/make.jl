using OptimaLib
using Documenter

DocMeta.setdocmeta!(
    OptimaLib,
    :DocTestSetup,
    :(using OptimaLib; import OptimaLib: solve);
    recursive = true,
)

makedocs(;
    clean    = false,
    modules  = [OptimaLib],
    authors  = "Jean-François Barthélémy",
    sitename = "OptimaLib.jl",
    remotes  = Dict(".." => Documenter.Remotes.GitHub("ChemistryTools", "OptimaLib.jl")),
    format   = Documenter.HTML(;
        canonical  = "https://ChemistryTools.github.io/OptimaLib.jl",
        edit_link  = "main",
        prettyurls = (get(ENV, "CI", nothing) == "true"),
        collapselevel = 1,
    ),
    pages = [
        "Home"            => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory"          => "theory.md",
        "Examples"        => [
            "Basic Usage"     => "examples/basic_usage.md",
            "Warm Start"      => "examples/warm_start.md",
            "Sensitivity"     => "examples/sensitivity.md",
            "SciML Interface" => "examples/sciml_interface.md",
        ],
        "API Reference"   => "api.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(; repo = "github.com/ChemistryTools/OptimaLib.jl", devbranch = "main")
end
