 #####################################################################################
 # lmg_driver.py                                                                     #
 #   Script for simulation of steady convection-diffusion heat problem on two-       #
 #   dimensional square domain using the Finite Volume Method on a cartesian,        #
 #   structured mesh with square cells. Central difference fluxes applied for        #
 #   the diffusive terms, and either central of upwinded difference fluxes           #
 #   applied for the convective terms.                                               #
 #   Implementation does not include source terms and is limited to the              #
 #   following sets of convective velocity fields and boundary conditions:           #
 #     (problem 1) see UniformFlow.py                                                #
 #     (problem 2) stagnation point flow [u,v] = [-x,y], homogeneous Neumann BC.     #
 #     at north wall (dTn/dy=0), inhomogeneous Dirichlet BC at west wall (Tw=0),     #
 #     inhomogeneous Dirichlet BC at east wall (Te=1), inhomogeneous Dirichlet BC    #
 #     at south wall (Ts=x).                                                         #
 #   Linear system of equations solved with linear multigrid solver using Galer-     #
 #   kin coarse grid smoother in recursive V-cycle.                                  #
 #                                                                                   #
 # Input         :                                                                   #
 #   K           :  Maximum number of grid level, implicitly defines grid size, n    #
 #   L           :  Size of square in x,y-direction                                  #
 #   Pe          :  Global Peclet number                                             #
 #   problem     :  Problem #: 1 or 2, selects case of convective field and BCs      #
 #   fvscheme    :  Finite volume scheme for convection-diffusion,                   #
 #                  either 'cds-cds' or 'uds-cds'                                    #
 #   w           :  Relaxation parameter for Jacobi-smoother.                        #
 #   imax        :  Maximum number of iterations for linsolver.                      # 
 #   nsmooth     :  Number of smoothening steps in lmg.                              # 
 #   tol         :  Relative tolerance for lmg.                                      #
 #                                                                                   #
 # Output        :                                                                   #
 #   T           :   Temperature at cell nodes, T(1:n²)                              #
 #   A           :   Convection-diffusion system matrix, A(1:n^2,1:n^2)              #
 #   s           :   Source array with BC contributions, s(1:n,1:n)                  #
 #   TT          :   Temperature field extrapolated to walls, TT(1:n+2,1:n+2)        #
 #   CF,DF       :   Conv. and diff. fluxes through walls, CF=[CFw,CFe,CFs,CFn]      #
 #   GHC         :   Global heat conservation, scalar (computed from wall fluxes)    #
 #   Plots of the temperature field and convective velocity field                    #
 #####################################################################################

import FVConvDiff2D
import numpy as np
from scipy.sparse import kron
import matplotlib.pyplot as plt
from matplotlib import cm
import lmg

plt.close('all')

## Input
K = 8  # max level
n = 2**(K+1)  # number of cells along x,y-axis
L = 1.0  # size of square in x,y-direction
Pe = 10.0  # global Peclet number
problem = 2  # problem to solve (1 or 2 - see file header)
fvscheme = 'cds-cds'  # finite volume scheme ('cds-cds' or 'uds-cds')
w = 0.6  # relaxation parameter for Jacobi-smoother
imax = 10000  # maximum number of iterations for linsolver
nsmooth = 3  # number of smoothening steps in lmg
tol = 1e-6  # relative tolerance for lmg

## MULTIGRID ##########################################################
# Predetermine grids and preassemble system matrices and store in dict'
# for lmg-solver

# Assemble system matrix
A, s = FVConvDiff2D.preprocess(n, L, Pe, problem, fvscheme)
s = s.reshape(n**2, order='F')

G = {}  # empty dictonary
for i in range(0, K):
    
    k = K-i  # grid level

    ni = 2**(k+1)  # number of grid points
    
    # 1D uniform restriction and prologoantion
    R, _ = lmg.restriction(ni)  # restriction for doubling mesh size
    P, _ = lmg.prolongation(ni)  # prolongation for halving mesh size.
    # extend to 2D using kronecker-product (see below)
    
    # Gallerkin coarse grid approximation
    if (k == K):
        G[k] = lmg.grid(n, A,
                        kron(R, R, format="csr"),
                        kron(P, P, format="csr"))
    else:
        G[k] = lmg.grid(n, G[k+1].R*G[k+1].A*G[k+1].P,
                        kron(R, R, format="csr"),
                        kron(P, P, format="csr"))

## get iterative solution
T, error, iter, flag = lmg.solve(np.ones(n**2), s, K,
                                 nsmooth, w, G, tol, imax)

#######################################################################

## Extend T-field to domain walls and get GHC-residual
TT, GHC, _, _ = FVConvDiff2D.postprocess(
    T.reshape(n, n, order='F'), n, L, Pe, problem, fvscheme)

## Plot solution and streamlines of the flow
plt.ion()  # turn on interactive mode
f, axarr = plt.subplots(1, 2, sharey=True, num=1)
f.suptitle('Convection-diffusion by %s for Pe = %d, \
flux-error = %0.3e' % (fvscheme, Pe, GHC))

# Coordinate arrays
dx = L/n  # cell size in x,y-direction
xf = np.arange(0., L+dx, dx)  # cell face coordinate vector along x,y-axis
xc = np.arange(dx/2., L, dx)  # cell center coordinates along x-axis
xt = np.hstack([0., xc, 1.])  # extended cell center coor. vector, incl. walls
Xc, Yc = np.meshgrid(xc, xc)  # cell center coordinate arrays
Xt, Yt = np.meshgrid(xt, xt)  # extended cell center coor. arrays, incl. walls

# Generate convective velocity field at cell faces
if problem == 1:  # problem 1 - uniform flow
    Uc = np.ones((np.size(Xc, 0), np.size(Xc, 1)))
    Vc = np.zeros((np.size(Xc, 0), np.size(Xc, 1)))
elif problem == 2:  # problem 2 - corner flow
    Uc = -Xc.copy()
    Vc = Yc.copy()
else:
    print('problem not implemented')

axarr[0].streamplot(  # only supports an evenly spaced grid
    xc, xc, Uc, Vc, density=1, linewidth=2,
    color=T.reshape(n, n, order='F'),
    cmap=cm.coolwarm, norm=None, arrowsize=1,
    arrowstyle='-|>', minlength=0.1)

axarr[0].set_title('Streamlines')
axarr[0].set_xlabel('x')
axarr[0].set_ylabel('y')
axarr[0].grid(True)
axarr[0].set_xlim(0, 1)
axarr[0].set_ylim(0, 1)

# Temperature field
p = axarr[1].pcolor(Xt, Yt, TT, cmap=cm.coolwarm, vmin=0, vmax=1)
axarr[1].set_title('Temperature')
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('y')
axarr[1].grid(True)
axarr[1].set_xlim(0, 1)
axarr[1].set_ylim(0, 1)
f.colorbar(p, ax=axarr[1])

f.show()
