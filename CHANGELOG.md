# Changelog

## [1.3.1] - 2026-10-22
### Added
- `FFBS_laplace!` and `laplace_kalmanfilter_update` take an optional user-supplied
  Fisher information `fisherinfo(θ, x, t)`, which replaces the observed information
  in the mode search and in the covariance at the mode
- Time-varying Poisson regression example (`examples/PoissonTvReg`)
### Changed
- The Laplace approximation passes its own ForwardDiff gradient and Hessian to Optim,
  which makes `FFBS_laplace!` about 16% faster on the Poisson example. Results are unchanged.

## [1.3.0] - 2026-10-19
### Removed
- `NewtonRaphson` (unused) and the old hand-rolled Newton mode finder
### Changed
- Fixed and completed the docstrings for the samplers and filter updates
- Docs now include the IPLF and Monte Carlo filter update docstrings
- Example manifests are no longer tracked, and the docs precompile the examples sequentially
- Codecov patch check disabled

## [1.2.0] - 2026-03-28
### Added
- Try-catch to make the FFBS sampler based on IPLF more robust
- Added a Monte Carlo version of the Gaussian approximation filters
- Created the CHANGELOG.md (this file)
- Added a .JuliaFormatter.toml for formatting the code.


## [1.1.4] - 2026-03-19
### Added
- Try-catch to make FFBS with laplace more robust
### Changed
- Refactored so the all FFBS samplers are mutating on the draws on container
- Changed the Backward sampling function to allocate less

[1.3.1]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.2.7...v1.3.0
[1.2.0]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.1.4...1.2.0
[1.1.4]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.1.3...v1.1.4