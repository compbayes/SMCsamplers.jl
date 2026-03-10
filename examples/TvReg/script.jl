# # Time-varying regression model

# In this example we explore the joint posterior of the state $\beta_t$ in a time-varying regression model with known static parameters: 
# ```math
# \begin{align*}
#   y_t &= z_t^\top\boldsymbol{\beta}_t + \varepsilon_t,  \quad \varepsilon_t \sim N(0,\sigma_\varepsilon) \\
#    \boldsymbol{\beta}_t &=  \boldsymbol{\beta}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(0,\boldsymbol{\Sigma}_\eta) \\
#   \boldsymbol{\beta}_0 &\sim N(0, \boldsymbol{\Sigma}_0)  
# \end{align*}
# ```

# Preliminaries

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, PDMats, LinearAlgebra
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate time-varying regression data
function SimTVReg(T, p, σₑ, Σₙ, Σ₀ = Σₙ)     
    Z = randn(T, p) # Matrix with covariates
    y = zeros(T)
    β = zeros(T, p)
    for t in 1:T
        if t == 1
            β₀ = rand(MvNormal(zeros(p), Σ₀))
            β[t,:] = β₀ + rand(MvNormal(Σₙ))
        else
            β[t,:] = β[t-1,:] + rand(MvNormal(Σₙ))
        end
        y[t] = Z[t,:] ⋅ β[t,:] + rand(Normal(0, σₑ))
    end
    return y, Z, β
end

p = 2      # State size - number of β parameters, including intercept
T = 200    # Number of observations
σₑ = 1
Σₙ = PDMat([1 0;0 0.1])     # State innovation covariance matrix
y, Z, β = SimTVReg(T, p, σₑ, Σₙ);

# Plot the parameter evolution
plt = plot()
for j in 1:p
    plot!(β[:,j]; label= L"\beta_{%$(j-1)}", c = colors[j], xlab ="t")
end
plt

# ### Set up time-varying regression models
mutable struct ParamTvReg
    σₑ::Float64
    Σₙ::PDMat{Float64}
    μ₀::Vector{Float64}
    Σ₀::PDMat{Float64}
    y::Vector{Float64} 
    Z::Matrix{Float64}
end
prior(θ) = MvNormal(θ.μ₀, θ.Σ₀)
transition(θ, state, t) = MvNormal(state, θ.Σₙ)
observation(θ, state, t) = Normal(θ.Z[t,:] ⋅ state, θ.σₑ)

μ₀ = zeros(p)
Σ₀ = PDMat(10*I(p))
θ = ParamTvReg(σₑ, Σₙ, μ₀, Σ₀, y, Z);

# ### PGAS
Nₚ = 20       # Number of particles
Nₛ = 1000     # Number of samples from posterior
PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation)
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = quantile_multidim(PGASdraws, [0.025, 0.975], dims = 3);

# ### FFBS
Σₑ = σₑ^2
A = collect(I(p))
C = zeros(1,p,T)
for t in 1:T
    C[:,:,t] = Z[t,:]
end
B = 0.0
U = zeros(T,1)

FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);
FFBSmean = mean(FFBSdraws, dims = 3)[2:end,:,1] # Exclude initial state at t=0
FFBSquantiles = quantile_multidim(FFBSdraws, [0.025, 0.975], dims = 3)[2:end,:,:];

# ### Plot the posterior mean and 95% credible intervals from both algorithms
plottrue = true
p = length(prior(θ))
plt = []
for j in 1:p
    ## True state evolution
    if plottrue
        plt_tmp = plot(β[:,j], c = colors[3], lw = 1, title = L"\beta_{%$(j-1)}", 
            label = "true state", legend = :bottomleft)
    else
        plt_tmp = plot(title = L"\beta_{%$(j-1)}", legend = :bottomleft)
    end
    ## PGAS
    plot!(PGASmean[:,j], lw = 1, c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 
    ## FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))

# The posterior mean and 95% credible intervals from both algorithms are indeed almost identical.