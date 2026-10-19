"""
    NewtonRaphson(∇, H, x0, tol = 1e-6, max_iter = 100, nugget = 1e-6; maximize = false)

Newton-Raphson method for finding an optimum of a function f with gradient ∇ and Hessian H.

Inputs:
- `∇`: gradient function, x -> ∇f(x)
- `H`: Hessian function, x -> ∇²f(x)
- `x0`: initial guess
- `tol`: convergence tolerance on the norm of the gradient
- `max_iter`: maximum number of iterations
- `nugget`: ridge on the Hessian diagonal for numerical stability. It is added,
  H(x) + nugget*I, when minimizing and subtracted, H(x) - nugget*I, when maximizing,
  so that it always moves the eigenvalues of H away from zero.
- `maximize`: set to true when f is maximized (H negative definite near the optimum)

Output:
- `x`: approximate optimum. Throws an error if not converged within `max_iter` iterations.
"""
function NewtonRaphson(∇, H, x0, tol = 1e-6, max_iter = 100, nugget = 1e-6;
    maximize = false)
    ridge = maximize ? -nugget : nugget
    x = x0
    for i in 1:max_iter
        grad = ∇(x)
        if norm(grad) < tol
            return x
        end
        x -= (H(x) + ridge*I) \ grad
    end
    error("Newton-Raphson did not converge within the maximum number of iterations")
end
