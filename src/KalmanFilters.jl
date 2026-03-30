""" 
    kalmanfilter_update(μ, Ω, u, y, A, B, C, Σₑ, Σₙ) 

A single Kalman filter update at time t of the state space model: 

yₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where
xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update(μ, Ω, u, y, A, B, C, Σₑ, Σₙ)

    # Prior propagation step - moving state forward without new measurement
    μ̄ = A*μ .+ B*u
    Ω̄ = A*Ω*A' + Σₙ

    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    K = Ω̄*C' / (C*Ω̄*C' .+ Σₑ) # Kalman Gain
    μ = μ̄ + K*(y .- C*μ̄)
    Ω = (I - K*C)*Ω̄
    return μ, Ω, μ̄, Ω̄

end


""" 
    kalmanfilter_update_extended(μ, Ω, u, y, A, B, C, ∂C, Cargs, Σₑ, Σₙ) 

A single extended Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update_extended(μ, Ω, u, y, A, B, C::Function, ∂C::Function, Cargs, Σₑ, Σₙ)

    # Prior propagation step
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ
    C̄ = ∂C(μ̄[:], Cargs) # y_t = C(θ_t) + ε_t

    # Measurement update
    K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ) 
    μ = μ̄ + K*(y .- C(μ̄[:], Cargs))
    Ω = (I - K*C̄)*Ω̄
    
    return μ, Ω, μ̄, Ω̄

end


""" 
    kalmanfilter_update_extended_iter(μ, Ω, u, y, A, B, C, ∂C, Cargs, Σₑ, Σₙ) 

A single extended Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update_extended_iter(μ, Ω, u, y, A, B, 
    C::Function, ∂C::Function, Cargs, Σₑ, Σₙ, maxIter = 1, tol = 1e-2)

    # Prior propagation step
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ

    # Measurement update
    μ_old = deepcopy(μ̄)             # overwritten later
    μ = deepcopy(μ̄) .+ 2*tol        # overwritten later
    i = 0
    while norm(μ .- μ_old) > tol && (i < maxIter)
        μ_old .= μ
        i += 1
        C̄ = ∂C(μ[:], Cargs) # y_t = C(θ_t) + ε_t
        K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ) 
        μ = μ̄ + K*(y .- C(μ[:], Cargs) - C̄*(μ̄ - μ))
    end
    #println("Extended Kalman filter converged after $i iterations") 
    if i > 1 # Recompute the gradient and Kalman gain to update the covariance matrix
        C̄ = ∂C(μ[:], Cargs) # y_t = C(θ_t) + ε_t
        K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ)
    end
    Ω = (I - K*C̄)*Ω̄
    
    return μ, Ω, μ̄, Ω̄

end


""" 
    kalmanfilter_update_extended_iter_line(μ, Ω, u, y, A, B, C, ∂C, Cargs, Σₑ, Σₙ) 

A single extended Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update_extended_iter_line(μ, Ω, u, y, A, B, 
    C::Function, ∂C::Function, Cargs, Σₑ, Σₙ, maxIter = 1, tol = 1e-2)

    # Prior propagation step
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ

    # Line search functions
    invΩ̄ = inv(Ω̄)
    invΣₑ = inv(Σₑ)
    V(μ) = (y - C(μ, Cargs))'*invΣₑ*(y - C(μ, Cargs)) + (μ-μ̄)'*invΩ̄*(μ-μ̄)
    Vprime(μ) = ForwardDiff.gradient(V, μ) 
    ϕ(α) = V(μ .+ α .* s) # line search function
    dϕ = α -> ForwardDiff.derivative(ϕ, α)
    ϕdϕ(α) = ϕ(α), dϕ(α)  # LineSearches.jl wants this tuple
    linesearch = BackTracking(order=3)

    # Measurement update
    Vval = V(μ̄)
    s = similar(μ̄) # search direction

    μ = μ̄ # initial value
    μ_old = deepcopy(μ) .+ 2*tol             # overwritten later
    i = 0
    while norm(μ .- μ_old) > tol && (i < maxIter)
        i += 1
        #println("Iteration $i, norm: $(norm(μ .- μ_old))")
        μ_old .= μ

        C̄ = ∂C(μ_old[:], Cargs) 
        K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ)
        
        # line search
        s = μ̄ - μ_old + K*(y .- C(μ_old[:], Cargs) - C̄*(μ̄ - μ_old)) # search direction
        α = 1.0 # initial step size
        
        Vval = V(μ)
        dϕval = s ⋅ Vprime(μ) 
        α, Vval = linesearch(ϕ, dϕ, ϕdϕ, α, Vval, dϕval)
        #println("Line search step size: $α, objective function value: $Vval")

        # Update 
        μ = μ_old + α*s

    end
    #println("Extended Kalman filter converged after $i iterations") 
    # Recompute the gradient and Kalman gain to update the covariance matrix
    C̄ = ∂C(μ[:], Cargs) # y_t = C(θ_t) + ε_t
    K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ)
    Ω = (I - K*C̄)*Ω̄
    
    return μ, Ω, μ̄, Ω̄

end


""" 
    kalmanfilter_update_unscented(μ, Ω, u, y, A, B, C, Cargs, Σₑ, Σₙ) 

A single unscented Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.4

"""
function kalmanfilter_update_unscented(μ, Ω, u, y, A, B, C::Function, Cargs, Σₑ, Σₙ, 
        γ, ωₘ, ωₛ)

    # Prediction step - moving state forward without new measurement
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ

    L̄ = cholesky(Ω̄).L # FIXME: This can be time-consuming, compute more efficiently?
    
    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    X̄ = [μ̄ (μ̄ .+ γ*L̄) (μ̄ .- γ*L̄)] # n×(2n+1) matrix with sigma points approx N(μ̄, Ω̄)
    x̂ = sum(X̄ .* ωₘ', dims = 2)
    Ȳ = mapslices(x -> C(x, Cargs), X̄, dims = 1) # Apply the C() function to each column
    ŷ = sum(Ȳ .* ωₘ', dims = 2)
    S = sum((ωₛ' .* (Ȳ .- ŷ))*(Ȳ .- ŷ)', dims = 2) .+ Σₑ
    Ω̄xy = sum((ωₛ' .* (X̄ .- μ̄))*(Ȳ .- ŷ)', dims = 2)
    K = Ω̄xy / S
    μ = μ̄ + K*(y .- ŷ)
    Ω = Hermitian(Ω̄ - K*S*K')

    return μ, Ω, μ̄, Ω̄

end


""" 
    kalmanfilter_update_IPLF(μ, Ω, u, y, A, B,  condMean, condCov, Cargs,  Σₙ, maxIter, γ, W) 

A single extended Kalman filter update at time t of the state space model: 

yₜ ~ p(xₜ),                               Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where f(xₜ) is the distribution of observations.

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 
condMean is the conditional mean of yₜ given xₜ
condCov is the conditional covariance of yₜ given xₜ

Reference: Simo Sarkka and Lennart Svensson (2023). Bayesian Filtering and Smoothing. Second Edition. Cambridge University Press.
"""

## PrLF and IPLF
function kalmanfilter_update_IPLF(μ, Ω, u, y, A, B, condMean, condCov, param,  Σₙ, t,
    maxIter, γ, ωₘ, ωₛ)

    ### Prior propagation
    μ̄ = A*μ .+ B*u
    Ω̄ = A*Ω*A' + Σₙ   

    μ = deepcopy(μ̄)
    Ω = deepcopy(Ω̄) 

    ### Measurement update
    for i in 1:maxIter

        L̄ = cholesky(Hermitian(Ω)).L

        ## Generate sigma points centered at the current mean μ
        χ = [μ μ .+ (L̄ * γ) μ .- (L̄ * γ)] # n×(2n+1) matrix with sigma points; N(μ, Ω)

        ## Propagate the sigma points through the conditional mean and covariance functions
        μₖ =  [condMean(param, χ[:, i], t) for i in 1:size(χ, 2)]
        Pₖʸ = [condCov(param, χ[:, i], t) for i in 1:size(χ, 2)]

        ## Compute the required moments:
        μₖ⁺ = sum(μₖ .* ωₘ) ## marginal mean of yₜ
        Δμ = reduce(hcat, [v .- μₖ⁺ for v in μₖ])' 
        Pₖˣʸ = (χ .- μ)* Diagonal(ωₛ) * Δμ # cross-covariance between xₜ and yₜ
        Pₖʸ = sum(ωₛ .* (Pₖʸ .+ [(v .- μₖ⁺)*(v .- μₖ⁺)' for v in μₖ])) # predicted measurement covariance
 
        ## Linearization of measurement model using Equations (10.14).
        # yₜ ≈ Aₖ * xₜ + bₖ + eₖ, where eₖ ~ N(0, Ωₖ)

        Aₖ = Pₖˣʸ' * inv(Ω)
        bₖ = μₖ⁺ .- Aₖ * μ
        Ωₖ = Pₖʸ .- Aₖ * Ω * Aₖ'
        
        ## Perform the Kalman update using the linearized model
        μₖⁱ = Aₖ * μ̄ + bₖ
        Sₖ = Aₖ * Ω̄ * Aₖ' + Ωₖ 
        Kₖ = try
            Ω̄ * Aₖ' / Sₖ 
        catch
            #println("Ω̄ = ");display(Ω̄)
            #println("Sₖ = ");display(Sₖ)
            #println("Aₖ = ");display(Aₖ)
            error("Kalman gain computation failed at iteration $i")
        end
        
        μ_updated = μ̄ + Kₖ * (y .- μₖⁱ)
        Ω_updated = Ω̄ - Kₖ * Sₖ * Kₖ'

        distance = KLD(μ, Ω, μ_updated, Ω_updated)
        
        if distance < 1e-3
            μ = μ_updated
            Ω = Ω_updated
            #println("Converged at iteration $i")
            break
        end

        μ = μ_updated
        Ω = Ω_updated
    end

    return μ, Ω, μ̄, Ω̄

end



""" 
    laplace_kalmanfilter_update(μ, Ω, u, y, A, B, observation, param, Σₙ, t) 


""" 
function laplace_kalmanfilter_update(μ, Ω, u, y, A, B, observation, param, Σₙ, t, 
        μ_init = nothing, max_iter=100)

    # Prior propagation step - moving state forward without new measurement
    μ̄ = A*μ .+ B*u
    Ω̄ = Hermitian(A*Ω*A' + Σₙ)

    if isnothing(μ_init) μ_init = μ̄  end

    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    μ, Ω = try
        filt_logpost(x) = logpdf(observation(param, x, t), y) + 
            logpdf(MvNormal(μ̄[:], Ω̄), x)
        laplace_approximation(filt_logpost, μ_init, 1.0, max_iter)  # Initial guess 
    catch
        error("Laplace approximation failed at time $t")
    end

    return μ, Ω, μ̄, Ω̄
end

function laplace_approximation(logposterior, initial_guess, cov_scale=1.0, max_iter=100)
    
    # Find mode (MAP estimate)
    handbaked = false
    if handbaked 
        function find_mode(x0)
            x = copy(x0)
            for _ in 1:max_iter
                g = ForwardDiff.gradient(logposterior, x)
                H = ForwardDiff.hessian(logposterior, x)
                #g = ForwardDiff.derivative(logposterior, x)
                #H = ForwardDiff.derivative(x -> ForwardDiff.derivative(logposterior, x), x)
                Δx = -H \ g  # Newton-Raphson step
                x += Δx
                if norm(Δx) < 1e-6
                    return x
                end
            end
            error("Mode finding did not converge")
        end
        θ_mode = find_mode(initial_guess)
    else
        optres = optimize(x -> -logposterior(x), initial_guess, 
            method = NewtonTrustRegion();
            autodiff = :forward, f_abstol = 1e-6, iterations = max_iter)
        θ_mode = Optim.minimizer(optres)
        if Optim.iterations(optres) > 10
            println("nIter to mode is larger than 10: $(Optim.iterations(optres))")
        end
    end
    
    # Compute Hessian at mode
    Σ = -inv(ForwardDiff.hessian(logposterior, θ_mode))  # Covariance matrix
    #Σ = -inv(ForwardDiff.derivative(θ_mode -> ForwardDiff.derivative(logposterior, #θ_mode), θ_mode))  # Covariance matrix
    
    # Adjust covariance if needed (sometimes too narrow/wide)
    Σ *= cov_scale^2
      
    # Return results
    return θ_mode, Σ

end


function kalmanfilter_update_montecarlo(μ, Ω, u, y, A, B, observation, param,  Σₙ, t, nMC)

    ### Prior propagation
    μ̄ = A*μ .+ B*u
    Ω̄ = A*Ω*A' + Σₙ   

    ### Measurement update

    ## Generate Monte Carlo draws from from prior propagated distribution N(μ̄, Ω̄)
    StateDraws = rand(MvNormal(μ̄[:], Hermitian(Ω̄)), nMC) # n × nMC matrix from N(μ̄, Ω̄)
    ObsDraws = zeros(length(y), nMC) # k × nMC matrix with sim observation data
    for j in 1:size(StateDraws, 2)
        ObsDraws[:,j] .= rand(observation(param, StateDraws[:,j], t)) 
    end

    ## Compute mean and covariance between state and measurements from MC draws
    μₒ = mean(ObsDraws, dims = 2)[:]
    Ωₒ = cov(ObsDraws')
    Ωₛₒ = cov(StateDraws', ObsDraws')
    K = try 
        Ωₛₒ / Ωₒ
    catch
        error("Kalman gain computation failed at time $t")
    end
    μ = μ̄ + K*(y .- μₒ)
    Ω = Ω̄ - K*Ωₒ*K' 

    return μ, Ω, μ̄, Ω̄

end