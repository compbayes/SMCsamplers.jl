using Distributions, LinearAlgebra, Random

@testset "Laplace filter with Fisher information" begin

    # Poisson regression with log link: canonical link, so the expected and observed
    # information coincide and both versions of the filter should give the same result.
    Random.seed!(1)
    T, n = 50, 2
    z = randn(T)
    y = rand.(Poisson.(exp.(1.0 .+ 0.5 * z)))
    θ = (z = z,)
    observation(θ, x, t) = Poisson(exp(x[1] + x[2] * θ.z[t]))
    function fisherinfo(θ, x, t)
        zt = [1.0, θ.z[t]]
        return exp(x[1] + x[2] * θ.z[t]) * zt * zt'
    end
    A = collect(1.0 * I(n))
    B = zeros(n)
    U = zeros(T)
    Σₙ = 0.01 * collect(1.0 * I(n))
    μ₀ = zeros(n)
    Σ₀ = collect(1.0 * I(n))
    Draws = zeros(T + 1, n)
    μ_obs, Σ_obs = FFBS_laplace!(Draws, U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ;
        filter_output = true)
    μ_fis, Σ_fis = FFBS_laplace!(Draws, U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ;
        filter_output = true, fisherinfo = fisherinfo)
    @test μ_fis ≈ μ_obs atol = 1e-6
    @test Σ_fis ≈ Σ_obs atol = 1e-6

    # Gaussian observation with nonlinear mean h(x) = x₁x₂: the Fisher information
    # J'J/σ² differs from the observed information, but the mode is the same and the
    # covariance is the Gauss-Newton one, inv(J'J/σ² + inv(Ω̄)).
    σ = 0.5
    obs_nonlin(θ, x, t) = Normal(x[1] * x[2], σ)
    fisher_nonlin(θ, x, t) = [x[2], x[1]] * [x[2], x[1]]' / σ^2
    μ = [1.0, 0.8]
    Ω = [0.3 0.1; 0.1 0.2]
    yt = 2.0
    μ_o, Ω_o, μ̄, Ω̄ = SMCsamplers.laplace_kalmanfilter_update(μ, Ω, 0.0, yt, A, B,
        obs_nonlin, nothing, Σₙ, 1)
    μ_f, Ω_f, _, _ = SMCsamplers.laplace_kalmanfilter_update(μ, Ω, 0.0, yt, A, B,
        obs_nonlin, nothing, Σₙ, 1, nothing, 100, fisher_nonlin)
    @test μ_f ≈ μ_o atol = 1e-5
    @test Ω_f ≈ inv(fisher_nonlin(nothing, μ_f, 1) + inv(Ω̄))
    @test !(Ω_f ≈ Ω_o)
end
