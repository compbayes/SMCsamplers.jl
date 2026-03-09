# Add local registry
using Pkg
Pkg.Registry.add(url="https://github.com/compbayes/CompBayesRegistry.git")

using SMCsamplers
using Test

include("SMCutilsTest.jl")

