 ##################################################################################
 # File: SolverEfforts.py                                                         #
 #   Script for obtaining and analyzing the computational costs related to        #
 #   solving the square system of linear equations A*T=s(:) of size n^2 using     #
 #   using dricet method or linear multigrid method.                              #
 #   Computations done with Pe = 10 and for both 'uds-cds' and 'cds-cds' schemes, #
 #   but limited to the the following convective velocity field and BC:           #
 #     1) uniform flow [u,v] = [1,0], homogeneous Neumann BC. at north and south  #
 #     walls (dTn/dy=dTs/dy=0), homogeneous Dirichlet BC at west wall (Tw=0),     #
 #     inhomogeneous Dirichlet BC at east wall (Te=1).                            #
 #                                                                                #
 # Input         :                                                                #
 #   N           :  Vector with number of cells along x,y-axis,                   #
 #                   test your code using e.g. N = [25 50 100 200],               #
 #                   but do your benchmarking using e.g. N = [100 200 400 800]    #
 #   L           :  Size of square in x,y-direction                               #
 #   Pe          :  global Peclet number                                          #
 #   problem     :  Problem , (set =1), selects case of convective field and BCs  #
 #   fvscheme    :  Finite volume scheme: 'cds-cds' or 'uds-cds'                  #
 #   OMEGAcoeff  :  Coeff. from analysis of optimal SOR relax. parameter for the  #
 #                  selected fvscheme, OMEGAcoeff = [c1 c2]                       #
 #   offset      :  Offset of omega from optimal value, used with small 'tol'     #
 #   TOL         :  Vector of relative tolerance on for iterative linear solver,  #
 #                  should be selected such that tol > condest(A)*eps, is         #
 #   imax        :  Maximum number of iterations for iterative linear solver      #
 #   linsolver   :  Linear solver (set ='sor')                                    #
 #                                                                                #
 # Output        :                                                                #
 #  - Plot of the computational costs (time) of solving A*T=s(:) w.r.t. the size  #
 #    of the problem, including the slopes of the curves (the scaling with size). #
 ##################################################################################

import FVConvDiff2D
import lmg
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron
from time import time

plt.close("all")

# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()
# # ... do something ...

## Input
Kvec = np.array([6, 7, 8])  # max level
N = 2**(Kvec+1)  # number of cells along x,y-axis
L = 1.0  # size of square in x,y-direction
Pe = 0.0  # global Peclet number
problem = 1  # problem to solve (set =1)
fvscheme = 'cds-cds'  # finite volume schemes analyzed
w = 0.6  # relaxation parameter for Jacobi-smoother
nsmooth = 3  # number of smoothening steps in lmg
TOL = np.array([1e-12, 1e-4])  # relative tolerance for linear solver
imax = 10000  # maximum iterations for linear solver

## Initialize
TIMEdirect = np.zeros(len(N))  # timing array, direct sol.
TIMElmg = np.zeros((len(TOL), len(N)))  # timing array, iter. sol.
ERROR = np.zeros((len(TOL), len(N), imax+1))
NUMiter = np.zeros((len(TOL), len(N)))
flag = np.zeros((len(TOL), len(N)))

## Time solving linear system of equations jj in range(0, len(N)):
for jj in range(0, len(N)):

    n = N[jj]
    K = Kvec[jj]

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
            G[k] = lmg.grid(ni, A,
                            kron(R, R, format="csr"),
                            kron(P, P, format="csr"))
        else:
            G[k] = lmg.grid(ni, G[k+1].R*G[k+1].A*G[k+1].P,
                            kron(R, R, format="csr"),
                            kron(P, P, format="csr"))

    #######################################################################

    # direct sparse solution
    start = time()
    T = spsolve(G[K].A, s)
    TIMEdirect[jj] = time() - start

    # lmg solution
    for ii in range(0, len(TOL)):    
        start = time()
        T, error, NUMiter[ii, jj], flag[ii, jj] = lmg.solve(np.ones(n**2), s, K,
                                                           nsmooth, w, G,
                                                           TOL[ii], imax)
        TIMElmg[ii, jj] = time() - start
        ERROR[ii, jj, 0:len(error)] = error



## TIME-scaling by power-law using polyfit
TLtime = np.zeros((len(N), len(TOL)+1))
Ptime = np.zeros((2, len(TOL)+1))
# direct
Ptime[:, 0] = np.polyfit(np.log(N**2), np.log(TIMEdirect), 1)
TLtime[:, 0] = np.polyval(Ptime[:, 0], np.log(N**2))
# iterative
for ii in range(1, len(TOL)+1):
    Ptime[:, ii] = np.polyfit(np.log(N**2), np.log(TIMElmg[ii-1, :]), 1)
    TLtime[:, ii] = np.polyval(Ptime[:, ii], np.log(N**2))


## ITERATION-scaling by power-law using polyfit
TLiter = np.zeros((len(N), len(TOL)))
Piter = np.zeros((2, len(TOL)))
# iterative
for ii in range(0, len(TOL)):
    Piter[:, ii] = np.polyfit(np.log(N**2), np.log(NUMiter[ii, :]), 1)
    TLiter[:, ii] = np.polyval(Piter[:, ii], np.log(N**2))


## Plot
plt.ion()  # turn on interactive mode

fig1 = plt.figure(1)
fig1.clf()

ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(N**2, TIMEdirect, 'o')
ax1.plot(N**2, TIMElmg.T, 'o')
ax1.plot(N**2, np.exp(TLtime), '-k')
ax1.legend(['DIRECT, slope=%0.2f' %Ptime[0, 0],
            'LMG, TOL=%0.2e, slope=%0.2f' %(TOL[0], Ptime[0, 1]),
            'LMG, TOL=%0.2e, slope=%0.2f' %(TOL[1], Ptime[0, 2])], loc=0)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('N^2')
ax1.set_ylabel('time')
ax1.grid(True)

fig1.show()

fig2 = plt.figure(2)
fig2.clf()

ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(N**2, NUMiter.T, 'o')
ax2.plot(N**2, np.exp(TLiter), '-k')
ax2.legend(['LMG, TOL=%0.2e, slope=%0.2f' %(TOL[0], Piter[0, 0]),
            'LMG, TOL=%0.2e, slope=%0.2f' %(TOL[1], Piter[0, 1])], loc=0)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('N^2')
ax2.set_ylabel('iterations')
ax2.grid(True)

fig2.show()

fig3 = plt.figure(3)
fig3.clf()

ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(range(1,imax+2), ERROR[:,-1,:].T, '-')
ax3.set_title('N=%i' %(N[-1]))
ax3.legend(['LMG, TOL=%0.2e, slope=%0.2f' %(TOL[0], Piter[0, 0]),
            'LMG, TOL=%0.2e, slope=%0.2f' %(TOL[1], Piter[0, 1])], loc=0)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('iterations')
ax3.set_ylabel('error')
ax3.grid(True)

fig3.show()


# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()
