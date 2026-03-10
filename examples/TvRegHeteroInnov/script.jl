# # Time-varying regression model with heteroscedastic innovations

# In this example we explore the joint posterior of the state $\beta_t$ in a time-varying regression model with known static parameters, but where the innovations in the states are heteroscedastic: 
# ```math
# \begin{align*}
#   y_t &= z_t^\top\boldsymbol{\beta}_t + \varepsilon_t,  \quad \varepsilon_t \sim N(0,\sigma_\varepsilon) \\
#   \boldsymbol{\beta}_t &= \boldsymbol{\beta}_{t-1} + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim N(0, \boldsymbol{Q}_t = \exp(\boldsymbol{h}_t)) \\
#   \boldsymbol{h}_t &= \boldsymbol{h}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(0,\boldsymbol{\Sigma}_\eta) \\
#   \boldsymbol{\beta}_0 &\sim N(\boldsymbol{0}, \boldsymbol{\Sigma_0})  
# \end{align*}
# ```
# We will take the $\boldsymbol{Q}_t = \exp(\boldsymbol{h}_t)$ sequence as known in this example.

# The example shows that PGAS can work poorly when a vague prior is used as the proposal distribution for the state at the first time step. This is the default setting in this package, but should not always be used. Later in the example we use a better proposal and show that PGAS works well.

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, PDMats, LinearAlgebra
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate time-varying regression data with heteroscedastic innovations
function SimTVReg(T, p, σₑ, Σₙ, μ = nothing, Φ = nothing, μ₀ = zeros(p), Σ₀ = Σₙ)
    Z = randn(T, p) # Matrix with covariates
    hetero = isnothing(μ) ? false : true
    #Simulate h-process and set up covariance matrices for the state innovations
    Q = [Σₙ for t = 1:T]
    if hetero
        H = repeat(μ', T, 1)
        Q[1] = PDMat(diagm(exp.(μ)))  
        for t = 2:T
            H[t,:] = μ + Φ*(H[t-1,:] - μ) + rand(MvNormal(Σₙ))
            Q[t] = PDMat(diagm(exp.(H[t,:]))) 
        end
    end
    y = zeros(T)
    β = zeros(T, p)
    for t in 1:T
        if t == 1
            β₀ = rand(MvNormal(μ₀, Σ₀))
            β[t,:] = β₀ + rand(MvNormal(Q[t]))
        else
            β[t,:] = β[t-1,:] + rand(MvNormal(Q[t]))
        end
        y[t] = Z[t,:] ⋅ β[t,:] + rand(Normal(0, σₑ))
    end
    return y, Z, β, Q
end

p = 2      # State size
T = 500    # Number of observations
σₑ = 1
μ = -8*ones(p)
Σₙ = PDMat(diagm(exp.(μ))) 
Φ = 0.8*I(p)
μ₀ = zeros(p)
Σ₀ = PDMat([100 0;0 100]) 
y, Z, β, Q = SimTVReg(T, p, σₑ, Σₙ, μ, Φ, μ₀, Σ₀)

#Plot the parameter evolution
plt = plot()
for j in 1:p
    plot!(β[:,j]; label= L"\beta_{%$(j-1)}", c = colors[j], xlab ="t")
end
plt

# ### Set up time-varying regression models
mutable struct ParametersTvRegHetero
    σₑ::Float64
    Q::Vector{PDMat{Float64}}
    μ₀::Vector{Float64}
    Σ₀::PDMat{Float64}
    y::Vector{Float64}
    Z::Matrix{Float64}
end
transition(θ, state, t) = MvNormal(state, θ.Q[t])
observation(θ, state, t) = Normal(θ.Z[t,:] ⋅ state, θ.σₑ)
prior(θ) = MvNormal(θ.μ₀, θ.Σ₀)
θ = ParametersTvRegHetero(σₑ, Q, μ₀, Σ₀, y, Z);  

# ### PGAS 
#First we use the prior as the proposal distribution for β₁ (which is the default)
Nₚ = 20       # Number of particles
Nₛ = 1000     # Number of samples from posterior
PGASdraws_prior = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation) 
PGASmean_prior = mean(PGASdraws_prior, dims = 3)[:,:,1]
PGASquantiles_prior = quantile_multidim(PGASdraws_prior, [0.025, 0.975], dims = 3);

#Plot update rates
update_rate = sum(abs.(diff(PGASdraws_prior[:,1,:]; dims = 2)) .> 0; dims=2) / Nₛ
plt1 = plot(update_rate; label=false, ylim=[0, 1], legend=:bottomleft, xlabel="Iteration",
    ylabel="Update rate", title = "PGAS with prior as proposal")
hline!([1 - 1 / Nₚ]; label="Optimal rate for N: $(Nₚ) particles", c = colors[3], 
    legend = :bottomright, lw = 1, linestyle = :dash)
plt1
# Note poor update rate for the early time steps. The prior is rather vague and is therefore not a good proposal for the state β₁.
# Let us run FFBS and the compare with the PGAS results

# ### FFBS
Σₑ = σₑ^2
A = collect(I(p))
C = zeros(1,p,T)
for t in 1:T
    C[:,:,t] = Z[t,:]
end
B = 0.0
U = zeros(T,1)
Σₙ = Q; 

FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);
FFBSmean = mean(FFBSdraws, dims = 3)[2:end,:,1] # Exclude initial state at t=0
FFBSquantiles = quantile_multidim(FFBSdraws, [0.025, 0.975], dims = 3)[2:end,:,:];

# The posterior from PGAS is poor and does not agree with the one from FFBS for the earlier time periods:

plottrue = true
p = length(prior(θ))
plt = []
for j in 1:p
    #True state evolution
    if plottrue
        plt_tmp = plot(β[:,j], c = colors[3], lw = 1, title = L"\beta_{%$(j-1)}", 
            label = "true state", legend = :bottom)
    else
        plt_tmp = plot(title = L"\beta_{%$(j-1)}", legend = :bottom)
    end
    #PGAS
    plot!(PGASmean_prior[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles_prior[:,j,1], fillrange = PGASquantiles_prior[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 
    #FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))

# The problem is that the prior is too vague to be a good proposal for the state at the first time step. Let's try a better proposal based on least squares fit to the first 20 obs
nInit = 20
μₚ = Z[1:nInit,:] \ y[1:nInit]
σ̂₀ = sqrt(sum((y[1:nInit] - Z[1:nInit,:]*μₚ).^2) / nInit)
Σₚ = PDMat(σ̂₀^2 * inv(Z[1:nInit,:]'*Z[1:nInit,:]))
initproposal(θ) = MvNormal(μₚ, Σₚ) # This is the proposal for the state β₁

PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation, initproposal); 
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = quantile_multidim(PGASdraws, [0.025, 0.975], dims = 3);

# The update rates are much better now
update_rate = sum(abs.(diff(PGASdraws[:,1,:]; dims = 2)) .> 0; dims=2) / Nₛ
plt2 = plot(update_rate; label=false, ylim=[0, 1], legend=:bottomleft, xlabel="Iteration",
    ylabel="Update rate", title = "PGAS with prior init proposal")
hline!([1 - 1 / Nₚ]; label="Optimal rate for N: $(Nₚ) particles", c = colors[3], 
    legend = :bottomright, lw = 1, linestyle = :dash)
plt2

# The inference from PGAS now agrees with the one from FFBS
plottrue = true
p = length(prior(θ))
plt = []
for j in 1:p
    #True state evolution
    if plottrue
        plt_tmp = plot(β[:,j], c = colors[3], lw = 1, title = L"\beta_{%$(j-1)}", 
            label = "true state", legend = :bottom)
    else
        plt_tmp = plot(title = L"\beta_{%$(j-1)}", legend = :bottom)
    end
    #PGAS
    plot!(PGASmean[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 
    #FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))