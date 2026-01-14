using LinearAlgebra
using LinearMapsAA
using Printf

# conjugate gradient solver function
function cg(A, x0, b; niter=30, tol=1e-6, verbose=false)

    # initialize variables
    t_start = time()
    r0 = b - A * x0
    if size(A, 1) != size(A, 2) # if non-square, solve normal equations
        r0 = A' * r0
    end
    rk = copy(r0)
    pk = copy(r0)
    xk = copy(x0)
    rktrk = real(dot(rk[:], rk[:]))
    t_end = time()
    if verbose
        println(@sprintf("cg initialization:  residual norm = %10.3e,  time = %8.3fs", sqrt(rktrk), t_end - t_start))
    end

    # loop through iterations
    for i in 1:niter
        t_start = time()

        # calculate step size
        Apk = A * pk
        if size(A, 1) != size(A, 2) # if non-square, solve normal equations
            Apk = A' * Apk
        end
        pktApk = dot(pk, Apk)
        alpha = rktrk / pktApk

        # gradient step
        xk1 = xk + alpha * pk
        rk1 = rk - alpha * Apk

        # update aux variables
        rk1trk1 = real(dot(rk1[:], rk1[:]))
        beta = rk1trk1 / rktrk
        pk1 = beta * pk + rk1

        # check for convergence
        if abs(sqrt(rktrk) - sqrt(rk1trk1)) < tol
            return xk
        end

        # update next iteration
        rk = rk1
        xk = xk1
        pk = pk1
        rktrk = rk1trk1

        t_end = time()
        if verbose
            println(@sprintf("cg iteration %4d:  residual norm = %10.3e,  time = %8.3fs", i, sqrt(rktrk), t_end - t_start))
        end

    end

    # return xk
    return xk

end

# function to compute eigenvectors of matrix using subspace iteration
function subspace_iteration(A, k; maxit=100, tol=1e-6)
    n = size(A, 1)
    V = randn(eltype(A), n, k) # initial random subspace
    V, _ = qr(V) # orthogonalize

    # subspace iteration
    for _ in 1:maxit
        V_new, _ = qr(A * V)
        if opnorm(V_new .- V, 2) < tol
            break
        end
        V = V_new
    end

    # get eigenvalues
    eigs = diag(V' * (A * V))
    return eigs, Matrix(V)
end