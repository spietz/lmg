import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, isspmatrix_csr


class grid():
    def __init__(self, size, sys, res, pro):
        self.n = size
        self.A = sys
        self.R = res
        self.P = pro


def my_diag(A):
    """
    Extraction of diag from a sparse matrix.
    """

    N = np.size(A, 1)  # get number of diag elements = num colums
    ii = np.arange(0, N)  # create seq of int from 1 to N
    return np.asarray(A[ii, ii]).flatten()  # ensure output is 1d array 


def smooth(x0, b, A, nsmooth, w):
    """
    Under-relaxed Jacobi smoother.
    """
    
    D = my_diag(A)
    x = x0

    for i in range(0, nsmooth):
        x = x + w*((b-A*x)/D)

    return x


def v_cycle(x0, b, k, nsmooth, w, G, tol):
    """
    Recursive V-cycle
    """

    if(k == 1):
        # direct solve
        xnew = spsolve(G[k].A, b)

        # return error and residual
        return xnew, 0  # note: no residual
    else:
        # pre-smooth
        x = smooth(x0, b, G[k].A, nsmooth, w)

        # calculate the residual
        r = b-G[k].A*x

        # coarsen the residual (k -> k-1)
        rc = G[k].R*r

        # recurse to Vcycle on a coarser grid (k-1)
        E, _ = v_cycle(np.zeros(len(rc)), -rc, k-1, nsmooth, w, G, tol)

        # project the solution from the coarse to the fine grid (k-1 -> k)
        E = G[k].P*E

        # update the error
        xnew = smooth(x-E, b, G[k].A, nsmooth, w)

        # return error and residual
        return xnew, r


def solve(x, b, K, nsmooth, w, G, tol, imax):
    """
    Linear Multigrid Solver

    Syntax:
    -------
    x,error,iter,flag = lmg.solve(x,b,K,nsmooth,w,G,tol,imax)

    Input:
    ------
    x0      :   initial guess for solution vector
    b       :   right hand side vector (make sure it is a vector!)
    K       :   number of grid-levels
    nsmooth :   number of Jacobi-smoothening operations at each level
    w       :   relaxation factor for Jacobi-smoother
    G       :   dictonary containing sparse grid operators
    tol     :   relative tolerance on residual

    Output:
    -------
    x       :  solution to A*x = b (converged)
    error   :  history of inf-norm of residual normed w.r.t. initial residual
    iter    :  number of iterations to reach converged solution
    flag    :  convergence flag (0: solution converged, 1: no convergence)
    """

    # Compute initial residual vector and norm
    returnflag = 0  # initialize flag for early function return
    rhsInorm = np.linalg.norm(b, np.inf)  # Inf-norm of rhs-vectorhon

    if rhsInorm == 0:  # homogene problem, x = b = 0
        x = b  # homogene solution
        error0 = 0  # zero error
        returnflag = 1  # return function
    else:  # inhomogene problem, non-zero rhs vector
        if np.linalg.norm(x, np.inf) == 0:  # zero initial guess
            res = rhsInorm  # norm of residual vector
            error0 = 1  # relative error
        else:  # non-zero initial guess
            r = b - G[K].A*x  # initial residual vector
            res = np.linalg.norm(r, np.inf)  # norm of residual vector
            error0 = res/rhsInorm  # relative error for initial guess
            if error0 <= tol:  # initial error less than tolerance
                returnflag = 1  # return function
        res0 = res  # ini. res. - stored for error computation

    # Return function
    if returnflag == 1:
        iter = 0
        flag = 0
        return x, error0, 0, flag

    # Iterate till error < tol
    iter = 0  # initialize iteration counter
    error = np.zeros(imax+1)  # vector to hold iteration error history
    error[0] = error0

    while iter < imax and error[iter] > tol:
        iter = iter+1  # update iteration counter
        x, r = v_cycle(x, b, K, nsmooth, w, G, tol)  # do a Vcycle
        res = np.linalg.norm(r, np.inf)  # norm of residual vector
        error[iter] = res/res0  # relative error

        # print status
        #print('Iteration: %i; relative residual: %e' %([iter, error(iter+1)]))

    error = error[0:iter+1]  # remove undone iterations from error

    # Check for final convergence
    if (error[iter] > tol):  # no convergence
        flag = 1  # failed convergence flag
    else:  # solution converged
        flag = 0  # convergence flag

        return x, error, iter, flag


def restriction(n_fine):
    """
    1d restriction operator:
    fine, 2^(k+1) -> coarse, 2^k
    """

    n_coarse = n_fine / 2
    if round(n_coarse) != n_coarse:
        print('Number of points/cells is not equal!')

    ## Uniform restriction operator
    # interpolation when doubling the mesh size
    I = np.zeros(4*(n_coarse-1))  # i-indexes (columns)
    J = np.zeros(4*(n_coarse-1))  # j-indexes (rows)
    S = np.zeros(4*(n_coarse-1))  # corresping (non-zero) values

    ## Fill in by rows:
    k = -1
    # linear interpolation at "left" endpoints
    I[k+1] = 0;        J[k+1] = 0;       S[k+1] = 1./2.
    I[k+2] = 0;        J[k+2] = 1;       S[k+2] = 1./2.
    k = k + 2

    # linear interpolation at interiour points (using average weighting)
    for j in range(2, n_coarse):
        I[k+1] = j-1; J[k+1] = 2*j-3; S[k+1] = 1./8.
        I[k+2] = j-1; J[k+2] = 2*j-2; S[k+2] = 3./8.
        I[k+3] = j-1; J[k+3] = 2*j-1; S[k+3] = 3./8.
        I[k+4] = j-1; J[k+4] = 2*j;   S[k+4] = 1./8.
        k = k + 4

    # linear interpolation at "right" endpoints
    I[k+1] = n_coarse-1; J[k+1] = n_fine-1; S[k+1] = 1./2.
    I[k+2] = n_coarse-1; J[k+2] = n_fine-2; S[k+2] = 1./2.

    # assemble sparse matrix
    R = csr_matrix((S, (I, J)), shape=(n_coarse, n_fine))
    return R, n_coarse


def prolongation(n_fine):
    """
    1d prologation operator
    coarse, 2^k -> fine, 2^(k+1)
    """

    n_coarse = n_fine / 2
    if round(n_coarse) != n_coarse:
        print('Number of points/cells is not equal!')

    ## Operator for linear uniform prolongation, P^k: T^k-1 -> T^k
    I = np.zeros(4*(n_coarse))  # i-indexes (columns)
    J = np.zeros(4*(n_coarse))  # j-indexes (rows)
    S = np.zeros(4*(n_coarse))  # corresping (non-zero) values

    k = -1
    # linear interpolation at "left" endpoints
    I[k+1] = 0;        J[k+1] = 0;          S[k+1] = 5./4.
    I[k+2] = 0;        J[k+2] = 1;          S[k+2] = -1./4.
    k = k + 2

    # linear interpolation at interiour points
    for j in range(1, n_coarse):
        I[k+1] = 2*j-1;   J[k+1] = j-1;   S[k+1] = 3./4.
        I[k+2] = 2*j-1;   J[k+2] = j;     S[k+2] = 1./4.
        I[k+3] = 2*j;     J[k+3] = j;     S[k+3] = 3./4.
        I[k+4] = 2*j;     J[k+4] = j-1;   S[k+4] = 1./4.
        k = k + 4

    # linear interpolation at "right" endpoints
    I[k+1] = n_fine-1; J[k+1] = n_coarse-2; S[k+1] = -1./4.
    I[k+2] = n_fine-1; J[k+2] = n_coarse-1; S[k+2] = 5./4.

    # assemble sparse matrix
    P = csr_matrix((S, (I, J)), shape=(n_fine, n_coarse))
    return P, n_coarse
