using LinearAlgebra

@testset "Misc.jl" begin

    # Quadratic f(x) = ±½(x - x̂)'Q(x - x̂) has its optimum at x̂
    Q = [2.0 0.5; 0.5 1.0]
    x̂ = [1.0, -2.0]
    x0 = zeros(2)

    # Minimization: f convex, H = Q positive definite
    xopt = NewtonRaphson(x -> Q * (x - x̂), x -> Q, x0)
    @test xopt ≈ x̂ atol = 1e-6

    # Maximization: f concave, H = -Q negative definite
    xopt = NewtonRaphson(x -> -Q * (x - x̂), x -> -Q, x0; maximize = true)
    @test xopt ≈ x̂ atol = 1e-6

    # Maximization with a nearly flat direction: the nugget must not flip its sign
    Qflat = Diagonal([1.0, 1e-7])
    ∇flat(x) = -Qflat * (x - x̂)
    xopt = NewtonRaphson(∇flat, x -> -Qflat, x0, 1e-12, 1000, 1e-6; maximize = true)
    @test xopt ≈ x̂ atol = 1e-4
    # Adding the nugget to a negative definite H flips that eigenvalue: no convergence
    @test_throws ErrorException NewtonRaphson(∇flat, x -> -Qflat, x0, 1e-12, 1000, 1e-6)

end
