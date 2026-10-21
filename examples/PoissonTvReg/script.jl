# # Time-varying Poisson regression

# In this example we analyze a Poisson regression with an intercept and one covariate
# where both regression coefficients evolve over time as independent Gaussian random walks:
# ```math
# \begin{align*}
#   y_t &\sim \mathrm{Poisson}\big(\exp(\beta_{0t} + \beta_{1t} z_t)\big), \quad z_t \sim N(0,1) \\
#   \beta_{jt} &= \beta_{j,t-1} + \nu_{jt}, \quad \nu_{jt} \sim N(0,\sigma_j^2), \quad j = 0,1 \\
#   \boldsymbol{\beta}_0 &\sim N(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
# \end{align*}
# ```
# The innovation variances are given independent scaled inverse chi-squared priors
# $\sigma_j^2 \sim \mathrm{Inv}\text{-}\chi^2(\nu_0, s_0^2)$ and the joint posterior of the
# state trajectories and the variances is explored by Gibbs sampling:
# 1. Draw the trajectories $\boldsymbol{\beta}_{0:T} = (\beta_{0t}, \beta_{1t})_{t=0}^T$ jointly from $p(\boldsymbol{\beta}_{0:T} \mid \sigma_0^2, \sigma_1^2, \boldsymbol{y})$.
# 2. Draw $\sigma_0^2$ from $p(\sigma_0^2 \mid \boldsymbol{\beta}_{0:T}, \boldsymbol{y})$.
# 3. Draw $\sigma_1^2$ from $p(\sigma_1^2 \mid \boldsymbol{\beta}_{0:T}, \boldsymbol{y})$.
#
# The full conditional of each variance is conjugate:
# ```math
#   \sigma_j^2 \mid \boldsymbol{\beta}_{0:T} \sim \mathrm{Inv}\text{-}\chi^2\left(\nu_0 + T,
#   \frac{\nu_0 s_0^2 + \sum_{t=1}^T (\beta_{jt} - \beta_{j,t-1})^2}{\nu_0 + T}\right).
# ```
# The trajectories in Step 1 are drawn with two different samplers, giving two separate Gibbs
# samplers that we compare: Particle Gibbs with ancestor sampling (PGAS) and FFBS with a
# Laplace approximation in each measurement update (FFBS_laplace, without state scaling).

# First some preliminaries:
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, PDMats, LinearAlgebra
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate data from the time-varying Poisson regression
function SimTvPoissonReg(T, σ², β₀)
    z = randn(T)
    β = zeros(T, 2)
    y = zeros(Int, T)
    for t in 1:T
        βprev = (t == 1) ? β₀ : β[t-1,:]
        β[t,:] = βprev + sqrt.(σ²) .* randn(2)
        y[t] = rand(Poisson(exp(β[t,1] + β[t,2]*z[t])))
    end
    return y, z, [β₀'; β]   # β includes the initial state at t = 0 in the first row
end

T = 500                     # Number of observations
nState = 2                  # State size (intercept and slope)
σ²true = [0.05^2, 0.03^2]   # True innovation variances
βinit = [1.5, 0.5]          # True initial state β₀
y, z, β = SimTvPoissonReg(T, σ²true, βinit)
timevect = 0:T;

plt1 = plot(timevect, β[:,1], label = L"\beta_{0t}", c = colors[1], xlab = "t",
    title = "parameter evolution")
plot!(timevect, β[:,2], label = L"\beta_{1t}", c = colors[2])
plt2 = scatter(1:T, y, label = "", markersize = 2, c = colors[3], xlab = "t",
    title = "counts")
plot(plt1, plt2, layout = (1,2), size = (800, 300))

# ### Set up the model structure for PGAS
# The struct is mutable so that the innovation variances can be updated in the Gibbs sampler.
mutable struct TvPoissonRegParams
    σ²::Vector{Float64}     # Innovation variances (σ₀², σ₁²)
    μ₀::Vector{Float64}
    Σ₀::PDMat{Float64}
    z::Vector{Float64}
end
prior(θ) = MvNormal(θ.μ₀, θ.Σ₀)
transition(θ, state, t) = MvNormal(state, Diagonal(θ.σ²))
observation(θ, state, t) = Poisson(exp(state[1] + state[2]*θ.z[t]));

# ### Prior settings
μ₀ = zeros(nState)              # Prior mean of β₀
Σ₀ = PDMat(4.0*I(nState))       # Prior covariance of β₀
ν₀ = 5                          # Prior degrees of freedom for σⱼ²
s₀² = 0.05^2                    # Prior best guess for σⱼ²
σ²init = [s₀², s₀²]             # Initial values of σⱼ² in the Gibbs sampler
θ = TvPoissonRegParams(copy(σ²init), μ₀, Σ₀, z);

# Gibbs update of a single innovation variance $\sigma_j^2$ given the trajectory of $\beta_{jt}$
function update_σ²(βj, ν₀, s₀²)
    T = length(βj) - 1
    νₙ = ν₀ + T
    sₙ² = (ν₀*s₀² + sum(abs2, diff(βj))) / νₙ
    return νₙ*sₙ² / rand(Chisq(νₙ))     # Draw from Inv-χ²(νₙ, sₙ²)
end

nIter = 3000                # Number of Gibbs iterations
nBurn = 500                 # Number of burn-in iterations to discard
keep = (nBurn+1):nIter;

# ### Gibbs sampling with PGAS for the state trajectories
# The prior $N(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ for $\boldsymbol{\beta}_0$ is rather
# vague and therefore a poor proposal in the particle filter. We instead use a proposal
# from a Laplace approximation of the posterior of a static Poisson regression on the first
# `nInit` observations, found by a few Newton steps.
nInit = 20
Zinit = [ones(nInit) z[1:nInit]]
μₚ = zeros(nState)
for _ in 1:20
    λ = exp.(Zinit*μₚ)
    μₚ .+= (Zinit'*Diagonal(λ)*Zinit + inv(Σ₀)) \ (Zinit'*(y[1:nInit] - λ) - Σ₀ \ μₚ)
end
Σₚ = PDMat(Symmetric(inv(Zinit'*Diagonal(exp.(Zinit*μₚ))*Zinit + inv(Σ₀))))
initproposal = MvNormal(μₚ, Σₚ);

nParticles = 50             # Number of particles for PGAS
X = zeros(nParticles, nState, T + 1)    # Particle buffer, including t = 0
PGASdraws = zeros(T + 1, nState, nIter)
PGASσ² = zeros(nIter, nState)
nFailurePGAS = Ref(0)
θ.σ² .= σ²init
βdraw = nothing             # No reference trajectory in the first iteration
for i in 1:nIter
    ## Step 1: Draw the trajectories of (β₀ₜ, β₁ₜ) jointly
    global βdraw = PGASsimulate!(X, y, nState, nParticles, θ, prior(θ), transition,
        observation, initproposal, systematic, βdraw; nFailure = nFailurePGAS)
    PGASdraws[:,:,i] = βdraw
    ## Steps 2-3: Draw σ₀² and σ₁² in separate updating steps
    θ.σ²[1] = update_σ²(βdraw[:,1], ν₀, s₀²)
    θ.σ²[2] = update_σ²(βdraw[:,2], ν₀, s₀²)
    PGASσ²[i,:] = θ.σ²
end
println("Number of PGAS failures: $(nFailurePGAS[])")

# The update rate of PGAS is the fraction of iterations where the state at a given time
# changes. The optimal rate with $N$ particles is $1 - 1/N$.
update_rate = sum(abs.(diff(PGASdraws[:,1,keep]; dims = 2)) .> 0; dims = 2) / length(keep)
plot(timevect, update_rate; label = false, ylim = [0, 1], xlabel = "t",
    ylabel = "Update rate", title = "PGAS update rate", legend = :bottomright)
hline!([1 - 1 / nParticles]; label = "Optimal rate for N: $(nParticles) particles",
    c = colors[3], lw = 1, linestyle = :dash)

# ### Gibbs sampling with FFBS_laplace for the state trajectories
# FFBS_laplace uses the observation density directly. The state equation is a random walk,
# so A = I and there are no controls.
A = collect(1.0*I(nState))
B = zeros(nState)
U = zeros(T)
FFBSdraws = zeros(T + 1, nState, nIter)
FFBSσ² = zeros(nIter, nState)
nFailureFFBS = Ref(0)
θ.σ² .= σ²init
βdraw = zeros(T + 1, nState)
for i in 1:nIter
    ## Step 1: Draw the trajectories of (β₀ₜ, β₁ₜ) jointly
    Σₙ = diagm(θ.σ²)
    FFBS_laplace!(βdraw, U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ;
        nFailure = nFailureFFBS)
    FFBSdraws[:,:,i] = βdraw
    ## Steps 2-3: Draw σ₀² and σ₁² in separate updating steps
    θ.σ²[1] = update_σ²(βdraw[:,1], ν₀, s₀²)
    θ.σ²[2] = update_σ²(βdraw[:,2], ν₀, s₀²)
    FFBSσ²[i,:] = θ.σ²
end
println("Number of FFBS_laplace failures: $(nFailureFFBS[])")

# ### Posterior of the state trajectories
# Posterior mean and 95% credible bands from both Gibbs samplers, after burn-in.
PGASmean = mean(PGASdraws[:,:,keep], dims = 3)[:,:,1]
PGASquantiles = quantile_multidim(PGASdraws[:,:,keep], [0.025, 0.975], dims = 3)
FFBSmean = mean(FFBSdraws[:,:,keep], dims = 3)[:,:,1]
FFBSquantiles = quantile_multidim(FFBSdraws[:,:,keep], [0.025, 0.975], dims = 3)

plt = []
for j in 1:nState
    plt_tmp = plot(timevect, β[:,j], c = colors[3], lw = 1, title = L"\beta_{%$(j-1)t}",
        label = "true state", legend = :bottom, xlab = "t")
    plot!(timevect, PGASmean[:,j], lw = 1, c = colors[j], label = "PGAS(N=$nParticles)")
    plot!(timevect, PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j], label = "", lw = 0)
    plot!(timevect, FFBSmean[:,j]; c = :black, lw = 1, linestyle = :dash,
        label = "FFBS_laplace")
    plot!(timevect, FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label = "")
    plot!(timevect, FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")
    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,nState), size = (800,300))

# ### Posterior of the innovation standard deviations
# Trace plots and posterior densities of $\sigma_j$ from both Gibbs samplers. The vertical
# lines in the histograms are the true values.
plt = []
for j in 1:nState
    plt_trace = plot(sqrt.(PGASσ²[:,j]), lw = 1, c = colors[j], label = "PGAS",
        title = L"\sigma_{%$(j-1)}" * " trace", xlab = "iteration", legend = :topright)
    plot!(sqrt.(FFBSσ²[:,j]), lw = 1, c = :black, alpha = 0.5, label = "FFBS_laplace")
    vline!([nBurn], c = :gray, lw = 1, linestyle = :dash, label = "")
    plt_hist = histogram(sqrt.(PGASσ²[keep,j]), normalize = :pdf, c = colors[j],
        alpha = 0.4, linecolor = :white, label = "PGAS",
        title = L"\sigma_{%$(j-1)}" * " posterior", legend = :topright)
    histogram!(sqrt.(FFBSσ²[keep,j]), normalize = :pdf, c = :gray, alpha = 0.4,
        linecolor = :white, label = "FFBS_laplace")
    vline!([sqrt(σ²true[j])], c = colors[3], lw = 2, label = "true")
    push!(plt, plt_trace, plt_hist)
end
plot(plt..., layout = (2,2), size = (800,500))
