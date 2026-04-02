# Changelog

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

[1.2.0]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.1.4...1.2.0
[1.1.4]: https://github.com/compbayes/SMCsamplers.jl/compare/v1.1.3...v1.1.4