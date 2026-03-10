module SMCsamplers

using Plots, LinearAlgebra, Distributions, Statistics, Random
using Measures, PDMats, LaTeXStrings, ForwardDiff
using LineSearches, Optim
using Utils: quantile_multidim
include("Misc.jl")
export NewtonRaphson

include("KalmanFilters.jl")
export kalmanfilter_update, kalmanfilter_update_extended, kalmanfilter_update_unscented
export kalmanfilter_update_extended_iter, kalmanfilter_update_extended_iter_line
export kalmanfilter_update_IPLF

include("FFBSsamplers.jl")
export FFBS, FFBSx, FFBS_unscented, FFBS_SLR, FFBS_laplace

include("ParticleGibbsSamplers.jl")
export PGASsimulate!, PGASsampler

include("SMCutils.jl")
export systematic, multinomial, ESS # TODO: add stratified resampling
export KLD, splitEqualGroups

end
  