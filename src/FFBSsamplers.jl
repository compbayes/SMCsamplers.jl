
# Backward sampling step for a given t. A is a vector or Matrix here.
function BackwardSim(x, μ_filt, Σ_filt, μ_pred, Σ_pred, A::AbstractArray, t)
    μback = μ_filt + Σ_filt*A'*( Σ_pred\(x .- μ_pred) )
    Σback = Σ_filt -  Σ_filt*A'*( Σ_pred \ A )*Σ_filt
    try 
        return rand(MvNormal(μback, Hermitian(Σback)))
    catch # Σₙ ≤ eps() so, Σ_pred ≈ Σ_filt and Σback ≈ 0. xₜ = xₜ₊₁ 
        return x   
    end
end

# Backward sampling step for a given t. A is a scalar here.
function BackwardSim(x, μ_filt, Σ_filt, μ_pred, Σ_pred, A::Number, t)
    μback = μ_filt + Σ_filt*A'*( Σ_pred\(x .- μ_pred) )
    Σback = Σ_filt -  Σ_filt*A'*( Σ_pred \ [A] )*Σ_filt
    try 
        return rand(MvNormal(μback, Hermitian(Σback)))
    catch # Σₙ ≤ eps() so, Σ_pred ≈ Σ_filt and Σback ≈ 0. xₜ = xₜ₊₁ 
        return x   
    end
end


function BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim = 1;  
    sample_t0 = true) # FIXME: this function assumes that A is static

    T, n = size(μ_filter)   # Number of time steps and state dimension

    # Backward sampling for t = T, T-1, ..., 1
    Xdraws = zeros(sample_t0 + T, n, nSim)  
    for i = 1:nSim 

        X = zeros(T, n) 
        X[T,:] = rand(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])))
        for t = (T-1):-1:1
            X[t,:] = BackwardSim(X[t+1,:], μ_filter[t,:], Σ_filter[:,:,t], μ_pred[t+1,:],   
                Σ_pred[:,:,t+1], A, t)
        end

        # Finally, sample state at t = 0
        if sample_t0
            x0 = BackwardSim(X[1,:], μ₀, Matrix(Σ₀), μ_pred[1,:], Σ_pred[:,:,1], A, 0)
            Xdraws[:,:,i] = [x0'; X]
        else
            Xdraws[:,:,i] = X
        end

    end

    if size(Xdraws, 3) == 1
        return Xdraws[:, :, 1] # Return a single draw as a T×n matrix
    else
        return Xdraws # Return all draws as a T×n×nIter array
    end
    return Xdraws

end



""" 
    Xdraws = FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, nSim = 1) 

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

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nIter.

""" 
function FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, nSim = 1; 
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

    Xdraws = BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim;        
        sample_t0 = sample_t0)

    if filter_output
        return Xdraws, μ_filter, Σ_filter
    else
        return Xdraws
    end

end




""" 
    FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

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

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nIter.

""" 
function FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀, nSim = 1, maxIter = 1, 
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

    Xdraws = BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim; 
        sample_t0 = sample_t0)
    if filter_output
        return Xdraws, μ_filter, Σ_filter
    else
        return Xdraws
    end

end

""" 
    FFBS_unscented(U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

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

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nIter.

""" 
function FFBS_unscented(U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀, nSim = 1; 
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

    Xdraws = BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim; 
        sample_t0 = sample_t0)

    if filter_output
        return Xdraws, μ_filter, Σ_filter
    else
        return Xdraws
    end

end


function FFBS_SLR(U, Y, A, B, condMean::Function, condCov::Function, param, Σₙ, μ₀, Σ₀,
        maxIter, nSim = 1; α = 1, β = 0, κ = 0, filter_output = false, 
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

    # Backward sampling for t = T, T-1, ..., 1
    Xdraws = BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim; 
        sample_t0 = sample_t0)

    if filter_output
        return Xdraws, μ_filter, Σ_filter
    else
        return Xdraws
    end

end



""" 
    FFBS_laplace(U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim = 1; filter_output = false) 

Forward filtering and backward sampling from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the general state space model:

yₜ ~ p(yₜ | xₜ)                     Measurement model

xₜ ~ p(xₜ | xₜ₋₁)                   State transition model

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

Note: If nSim == 1, the returned Xdraws is matrix, otherwise it is a 3D array of size T×n×nIter.

""" 
function FFBS_laplace(U, Y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim = 1; 
    filter_output = false, sample_t0 = true, μ_init = nothing, max_iter = 100)

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
        μ, Σ, μ̄, Σ̄ = laplace_kalmanfilter_update(μ, Σ, u, Y[t], At, B, 
            observation, θ, Σₙt, t, μ_init, max_iter)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    Xdraws = BackwardSampling(μ_filter, Σ_filter, μ_pred, Σ_pred, A, μ₀, Σ₀, nSim; 
        sample_t0 = sample_t0)

    if filter_output
        return Xdraws, μ_filter, Σ_filter
    else
        return Xdraws
    end

end


