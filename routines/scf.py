#!/usr/bin/python

from __future__ import division
import math
import numpy as np
import routines.ints as ints
from numpy import genfromtxt
from utils import printmat
#import matplotlib.pyplot as plt


# Symmetrize a matrix given a triangular one
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


# Return compund index given four indices
def eint(a,b,c,d):
    if a > b:
        ab = a*(a+1)/2 + b
    else:
        ab = b*(b+1)/2 + a
    if c > d:
        cd = c*(c+1)/2 + d
    else:
        cd = d*(d+1)/2 + c
    if ab > cd:
        abcd = ab*(ab+1)/2 + cd
    else:
        abcd = cd*(cd+1)/2 + ab
    return abcd


# Return Value of two electron integral
# Example: (12|34) = tei(1,2,3,4)
# I use chemists notation for the SCF procedure. 
def tei(a,b,c,d,twoe):
    return twoe.get(eint(a,b,c,d),0.0)


# Put Fock matrix in Orthonormal AO basis
def fprime(X,F):
    return np.dot(np.transpose(X),np.dot(F,X))


# Make Density Matrix
def makedensity(C,P,dim,Nelec):
    OLDP = np.zeros((dim,dim))
    for mu in range(0,dim):
        for nu in range(0,dim):
            OLDP[mu,nu] = P[mu,nu]
            P[mu,nu] = 0.0e0
            for m in range(0,Nelec//2):
                P[mu,nu] = P[mu,nu] + 2*C[mu,m]*C[nu,m]
    return P, OLDP


# Make Fock Matrix
# The most naive implementation, better one provided in minichem project
def makefock(Hcore, P, dim, twoe):

    F = np.zeros((dim, dim))

    for i in range(0, dim):
        for j in range(0, dim):
          F[i, j] = Hcore[i, j]
          for k in range(0, dim):
              for l in range(0, dim):
                  F[i, j] += P[k,l]*(tei(i+1,j+1,k+1,l+1,twoe) - 0.5*tei(i+1,k+1,j+1,l+1,twoe))

    return F 


# Calculate change in density matrix
def deltap(P,OLDP):
    DELTA = 0.0e0
    for i in range(0,dim):
        for j in range(0,dim):
            DELTA += (P[i,j] - OLDP[i,j])**2
    DELTA = (DELTA/4)**(0.5)
    return DELTA


# Calculate change in energy
def deltae(E,OLDE):
    return abs(E-OLDE)
# Calculate energy at iteration
def currentenergy(P,Hcore,F,dim):
    EN = 0.0
    for mu in range(0,dim):
        for nu in range(0,dim):
            EN += 0.5*P[mu,nu]*(Hcore[mu,nu] + F[mu,nu])
    return EN


def print_mos(E,C,Nelec):
    print '\n\tRHF final molecular orbitals'
    print '\t----------------------------'
    print '  N      E\tOcc\tCoefficients'
    dim = len(C)
    for i in xrange(0,dim):
        print "%3d%10.4f%5d" % (i+1, E[i], (2 if (i < Nelec/2) else 0)),
        for j in xrange(0,dim):
            print "%10.4f" % C[j,i],
        print
    print


def mulliken(PS,geom,bfns):
    def double_vec_eq(a, b):
        for i in xrange(0,len(a)):
            if abs(a[i]-b[i]) >= 1e-14:
                return False
        return True

    print '\n\tMulliken Population Analysis'
    print '\t----------------------------\n'
    print '   Z      x        y        z       N(elec)   Charge'
    print '-------------------------------------------------------'
    for i,at in enumerate(geom):
        Z, x, y, z = at
        pop = 0.0
        for j, b in enumerate(bfns):
            if double_vec_eq([x,y,z],b.origin):
                pop += PS[j,j]
        qA = Z - pop
        print '%4d%9.4f%9.4f%9.4f | %9.4f%9.4f' % (Z, x, y, z, pop, qA)
    print '-------------------------------------------------------'
    print


def scf_iteration(convergence,ENUC,Nelec,dim,S,Hcore,twoe,printops,do_DIIS,geom,bfns):
    ######################################
    #
    #   SCF PROCEDURE
    #
    #####################################

    # Step 1: orthogonalize the basis (I used symmetric orthogonalization,
    # which uses the S^(-1/2) as the transformation matrix. See Szabo and Ostlund
    # p 143 for more details.

    print 'Basis set orthogonalization'
    SVAL, SVEC = np.linalg.eigh(S)
    print 'min S eigenvalue = ', np.min(SVAL)
    SVAL_minhalf = (np.diag(SVAL**(-0.5)))
    X = S_minhalf = np.dot(SVEC,np.dot(SVAL_minhalf,np.transpose(SVEC)))

    # This is the main loop. See Szabo and Ostlund p 146. 
    # Try uncommenting the print lines to see step by step 
    # output. Look to see the functions defined above for a better
    # understanding of the process. 
    if do_DIIS == True:
        print "DIIS acceleration ON"
    elif do_DIIS == False:
        print "DIIS acceleration OFF"
    P = np.zeros((dim,dim)) # P is density matrix, set intially to zero.
    OLDE = 0.0
    num_e = 6
    ErrorSet = []
    FockSet = []
    for j in range(0,120):    # maxiter = 120
        F = makefock(Hcore,P,dim,twoe)
        if do_DIIS == True:
            if j > 0:
                error = ((np.dot(F,np.dot(P,S)) - np.dot(S,np.dot(P,F))))
                if len(ErrorSet) < num_e:
                    FockSet.append(F)
                    ErrorSet.append(error)
                elif len(ErrorSet) >= num_e:
                    del FockSet[0]
                    del ErrorSet[0]
                    FockSet.append(F) 
                    ErrorSet.append(error)
            NErr = len(ErrorSet)
            if NErr >= 2:
                Bmat = np.zeros((NErr+1,NErr+1))
                ZeroVec = np.zeros((NErr+1))
                ZeroVec[-1] = -1.0
                for a in range(0,NErr):
                    for b in range(0,a+1):
                        Bmat[a,b] = Bmat[b,a] = np.trace(np.dot(ErrorSet[a].T,ErrorSet[b]))
                        Bmat[a,NErr] = Bmat[NErr,a] = -1.0
                try:
                    coeff = np.linalg.solve(Bmat,ZeroVec)
                except np.linalg.linalg.LinAlgError as err:
                    if 'Singular matrix' in err.message:
                        print '\tSingular Pulay matrix, turing off DIIS'
                        do_DIIS = False
                else:
                    F = 0.0
                    for i in range(0,len(coeff)-1):
                        F += coeff[i]*FockSet[i]

        # Put Fock in orthonormal AO
        Fprime = fprime(X,F)
        # Solve F'C' = eC'
        E,Cprime = np.linalg.eigh(Fprime)
        # C back to AO basis
        C = np.dot(X,Cprime)
        P,OLDP = makedensity(C,P,dim,Nelec)
        # test for convergence. if meets criteria, exit loop and calculate properties of interest
        DELTA = deltae(currentenergy(P,Hcore,F,dim),OLDE)
        if printops == True:
	        print "E: {0:.12f}".format(currentenergy(P,Hcore,F,dim)+ENUC),"a.u.",'\t',"Delta: {0:.12f}".format(DELTA)
        OLDE = currentenergy(P,Hcore,F,dim)
        if DELTA < convergence:
            print "NUMBER ITERATIONS: ",j
            #printmat("Final Fock matrix in AO basis", F)
            FMO = np.dot(np.transpose(C), np.dot(F, C))
            #printmat("Final Fock matrix in MO basis", FMO)
            #print_mos(E,C,Nelec)
            # mulliken
            PS = np.dot(P,S)
            mulliken(PS,geom,bfns)
            #printmat("Mulliken analysis matrix PS", PS)
            print 'Nelec = ', np.trace(PS)
            break
        if j == 119:
            print "SCF not converged!"
            break
        EN = currentenergy(P,Hcore,F,dim)
    #print "TOTAL E(SCF) = \n", EN + ENUC
    #print "C = \n", C
    return EN, E, C, P, F



