using nonlinearlstr
using Documenter

DocMeta.setdocmeta!(nonlinearlstr, :DocTestSetup, :(using nonlinearlstr); recursive=true)

makedocs(;
    modules=[nonlinearlstr],
    authors="Your Name <your.email@example.com> and contributors",
    sitename="nonlinearlstr.jl",
    format=Documenter.HTML(;
        canonical="https://vcantarella.github.io/nonlinearlstr",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/vcantarella/nonlinearlstr",
    devbranch="main",
)