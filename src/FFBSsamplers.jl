# nSim = 1 method: Xdraws is (T + sample_t0) × n matrix
@views function BackwardSampling!(Xdraws::AbstractMatrix, μ_filter, Σ_filter, 
        μ_pred, Σ_pred, A, μ₀, Σ₀; sample_t0 = true)

    T, n = size(μ_filter)
    x     = zeros(n)
    μback = zeros(n)
    μ_zero = zeros(n)
    AS    = zeros(n, n)
    G     = zeros(n, n)

    rand!(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])), x)
    Xdraws[T + sample_t0, :] .= x

    for t = (T-1):-1:1
        mul!(AS, A, Σ_filter[:,:,t])
        G    .= (Σ_pred[:,:,t+1] \ AS)'
        Σback = Hermitian(Σ_filter[:,:,t] - G * AS)
        μback .= μ_filter[t,:] .+ G * (Xdraws[t+1+sample_t0,:] .- μ_pred[t+1,:])
        try
            rand!(MvNormal(μ_zero, Σback), x)
            Xdraws[t+sample_t0,:] .= μback .+ x
        catch
            Xdraws[t+sample_t0,:] .= Xdraws[t+1+sample_t0,:]
        end
    end

    if sample_t0
        Σ₀mat = Matrix(Σ₀)
        mul!(AS, A, Σ₀mat)
        G    .= (Σ_pred[:,:,1] \ AS)'
        Σback = Hermitian(Σ₀mat - G * AS)
        μback .= μ₀ .+ G * (Xdraws[2,:] .- μ_pred[1,:])
        try
            rand!(MvNormal(μ_zero, Σback), x)
            Xdraws[1,:] .= μback .+ x
        catch
            Xdraws[1,:] .= Xdraws[2,:]
        end
    end
end

# Thin wrapper: loop over sims and delegate to 2D
# Not used, but may try it to avoid code duplication. Slower though
function BackwardSamplingThin!(Xdraws::AbstractArray{<:Real,3}, μ_filter, Σ_filter, 
        μ_pred, Σ_pred, A, μ₀, Σ₀, nSim = 1; sample_t0 = true)
    for i = 1:nSim
        BackwardSampling!(@view(Xdraws[:,:,i]), μ_filter, Σ_filter, μ_pred, Σ_pred, A, 
            μ₀, Σ₀; sample_t0 = sample_t0)
    end
end

# nSim > 1 method: Xdraws is (T + sample_t0) × n × nSim array — dispatch on AbstractArray
@views function BackwardSampling!(Xdraws::AbstractArray{<:Real,3}, μ_filter, Σ_filter, 
        μ_pred, Σ_pred, A, μ₀, Σ₀; sample_t0 = true)
    
    T, n = size(μ_filter)   # T does not include t=0, n is the dim of state
    nSim = size(Xdraws, 3)  # number of draws
    X = zeros(n, nSim)      # buffer, reused every t
    μback = zeros(n, nSim)
    μ_zero = zeros(n)       # pre-allocated zero mean for MvNormal
    AS = zeros(n, n)        # buffer for A * Σ_filter
    G  = zeros(n, n)        # backward gain matrix

    # Sample all nSim iter at once at t = T
    rand!(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])), X) # nSim iid draws
    Xdraws[T + sample_t0, :, :] .= X

    # Backward sampling for t = T-1, ..., 1
    for t = (T-1):-1:1
        mul!(AS, A, Σ_filter[:,:,t])
        G .= (Σ_pred[:,:,t+1] \ AS)'
        Σback = Hermitian(Σ_filter[:,:,t] - G * A * Σ_filter[:,:,t])
        μback .= μ_filter[t,:] .+ G * (Xdraws[t+1+sample_t0,:,:] .- μ_pred[t+1,:])# n × nSim
        try
            rand!(MvNormal(μ_zero, Σback), X) # exploit that Σback not a function of state
            Xdraws[t+sample_t0,:,:] .= μback .+ X
        catch
            Xdraws[t+sample_t0,:,:] .= Xdraws[t+1+sample_t0,:,:]
        end
    end

    # Sample at t = 0
    if sample_t0
        Σ₀mat = Matrix(Σ₀)   # outside would be better 
        mul!(AS, A, Σ₀mat)
        G .= (Σ_pred[:,:,1] \ AS)'
        Σback = Hermitian(Σ₀mat - G * AS)
        μback = μ₀ .+ G * (Xdraws[2,:,:] .- μ_pred[1,:])
        try
            rand!(MvNormal(μ_zero, Σback), X)
            Xdraws[1,:,:] .= μback .+ X
        catch
            Xdraws[1,:,:] .= Xdraws[2,:,:]
        end
    end

end




""" 
    Xdraws = FFBS!(Draws, U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, nSim = 1) 

Forward filtering and backward sampling from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model:

yₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀.
A, C, Σₑ and Σₙ can be deterministically time-varying by passing 3D arrays of size n×n×T.

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nSim.

""" 
function FFBS!(Draws, U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀; 
        filter_output = false, sample_t0 = true)

    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticC = (ndims(C) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Ct = staticC ? C : @view C[:,:,t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        #y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update(μ, Σ, u, Y[t], At, B, Ct, Σₑt, Σₙt)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; 
        sample_t0 = sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing

end




""" 
    FFBSx!(Draws, U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

Forward filtering and backward sampling from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

C(xₜ) is a non-linear function that we can ForwardDiff.jl to get the Jacobian

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nSim.

""" 
function FFBSx!(Draws, U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀, maxIter = 1, 
    tol = 1e-2, linesearch = false; filter_output = false, sample_t0 = true)

    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true
    staticCargs = (ndims(Cargs) == 3 || eltype(Cargs) <: Vector) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Cargs_t = staticCargs ? Cargs : Cargs[t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        #y = (r == 1) ? Y[t] : Y[t,:]
        if maxIter == 1
            μ, Σ, μ̄, Σ̄ = kalmanfilter_update_extended(μ, Σ, u, Y[t], At, B, C, ∂C, Cargs_t, Σₑt, Σₙt)
        else 
            if linesearch 
                μ, Σ, μ̄, Σ̄ = kalmanfilter_update_extended_iter_line(μ, Σ, u, Y[t], At, B, C, ∂C, Cargs_t, Σₑt, Σₙt, maxIter, tol)
            else
                μ, Σ, μ̄, Σ̄ = kalmanfilter_update_extended_iter(μ, Σ, u, Y[t], At, B, C, ∂C, Cargs_t, Σₑt, Σₙt, maxIter, tol)
            end
        end
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; 
        sample_t0 = sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing

end

""" 
    FFBS_unscented!(Draws, U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

Forward filtering and backward sampling from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

C(xₜ) is a non-linear function

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nSim.

""" 
function FFBS_unscented!(Draws, U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀; 
        α = 1, β = 0, κ = 0, filter_output = false, sample_t0 = true)

    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Set up the weights for the unscented Kalman filter
    λ = α^2*(n + κ) - n # λ = 3-n # me = 1
    ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
    ωₛ = [λ/(n + λ) + (1 - α^2 + β); ωₘ[2:end]]
    γ = sqrt(n + λ) # Ganna: sqrt(3) # sqrt(n + 1)

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Cargs_t = Cargs[t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        #y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update_unscented(μ, Σ, u, Y[t], At, B, C, Cargs_t, 
            Σₑt, Σₙt, γ, ωₘ, ωₛ)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; 
        sample_t0 = sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing

end


""" 
    FFBS_SLR!(Draws, U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim = 1; 
        filter_output = false, sample_t0 = true) 

Forward filtering using Statistical Linear Regression for linearizing, followed by backward sampling from the joint smoothing posterior: 
p(x1,...xT | y1,...,yT) of the general state space model:

yₜ ~ p(yₜ | xₜ)                     Measurement model

xₜ ~ q(xₜ | xₜ₋₁)                   State transition model

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nSim.

""" 
function FFBS_SLR!(Draws, U, Y, A, B, condMean::Function, condCov::Function, param, Σₙ, 
        μ₀, Σ₀, maxIter; α = 1, β = 0, κ = 0, filter_output = false, 
        sample_t0 = true)
    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Set up the weights for the UT transform
    λ  = α^2*(n + κ) - n
    ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
    ωₛ  = [λ/(n + λ) + (1 - α^2 + β); ωₘ[2:end]]

    γ = sqrt(n + λ)

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)

    for t = 1:T 
        At = staticA ? A : @view A[:,:,t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update_IPLF(μ, Σ, u, Y[t], At, B, condMean, condCov, 
            param,  Σₙt, t, maxIter, γ ,ωₘ, ωₛ)

        #println("Time step: ", t, " Mean: ", μ, " Covariance: ", Σ)
        
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; 
        sample_t0 = sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing

end



""" 
    FFBS_laplace!(Draws, U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim = 1; 
        filter_output = false) 

Forward filtering and backward sampling from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the general state space model:

yₜ ~ p(yₜ | xₜ)                     Measurement model

xₜ ~ p(xₜ | xₜ₋₁)                   State transition model

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nSim.

""" 
function FFBS_laplace!(Draws, U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ; 
    filter_output = false, sample_t0 = true, μ_init = nothing, max_iter = 100,
    nFailure = Ref(0))

    T = length(Y)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    #r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        #y = (r == 1) ? Y[t] : Y[t,:]
        filter_result = try
            laplace_kalmanfilter_update(μ, Σ, u, Y[t], At, B, observation, θ, Σₙt, t, 
                μ_init, max_iter)
        catch
            nFailure[] += 1
            return nothing
        end
        μ, Σ, μ̄, Σ̄ = filter_result
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    BackwardSampling!(Draws, μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀; 
        sample_t0 = sample_t0)

    if filter_output
        return μ_filter, Σ_filter
    end
    return nothing

end


