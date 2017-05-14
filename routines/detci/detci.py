#!/usr/bin/python

from __future__ import division
import math
import time
import numpy as np

########################################################################
#
#   Determinant-based configuration interaction singles (CIS)
#
########################################################################

# E0 - Hartree-Fock reference energy
# f - Fock matrix in spinorbital basis
# ints - two-electron intagrals in spinorbital basis
def cis(E0,f,ints,Nelec,dim):
    print '\n\tConfiguration Interaction Singles'
    print '\t---------------------------------'
    print 'Reference type: RHF'
    print 'Reference energy E0 = ', E0
    size = Nelec*(2*dim - Nelec) # number occupied * number virtual
    printall = False
    print 'Problem size: %dx%d' % (size,size)
    
    # Construction of Singles-Singles Block of normal-ordered Hamiltonian matrix
    t0 = time.time()
    HSS = np.zeros((size,size))
    I = -1
    for i in range(0,Nelec):
        for a in range(Nelec,dim*2):
            I = I + 1
            J = -1
            for j in range(0,Nelec):
                for b in range(Nelec,dim*2):
                    J = J + 1
                    HSS[I,J] = -(a==b)*f[i,j] + (i==j)*f[a,b] - ints[j, a, i, b]
    print '<S|H|S> constructed in %.3f sec' % (time.time()-t0)
    
    # HN = ( 0     0    )
    #      ( 0  <S|H|S> )
    (dE,civecs) = np.linalg.eigh(HSS)
    
    if size < 10:
        printall = True
    if printall == True:
        print 'Configurations:'
        print '---------------'
        I = 0
        for i in range(0,Nelec):
            for a in range(Nelec,dim*2):
                I = I + 1
                print '%d\t(' % (I),
                for k in range(0,dim*2):
                    if (k != i and k < Nelec) or (k == a and k >= Nelec):
                        print 1,
                    else:
                        print 0,
                print ')'
                            
        print 'Roots:'
        print '------'
        print '  N\tdE\t\tCoefficients'
        
        
        for i in range(0, size):
            print "%3d%10.4f" % (i+1, dE[i]),
            for j in range (0,size):
                print "%10.4f" % civecs[j,i],
            print
        print
    else:
        print 'Roots    a.u.        cm^-1:'
        print '---------------------------'
        for i in range(0, size):
            print "%3d%10.4f%14.2f" % (i+1, dE[i], dE[i]*219474.63)
    del HSS

########################################################################
#
#   Determinant-based configuration interaction doubles (DET-CID)
#
########################################################################

def cid(E0,f,ints,Nelec,dim):
    print '\n\tConfiguration Interaction Doubles (DCI)'
    print '\t---------------------------------------'
    print 'Reference type: RHF'
    print 'Reference energy E0 = ', E0
    print 'Nelec = ', Nelec
    dim = dim*2
    print 'dim = ', dim
    No = Nelec
    Nv = dim - Nelec
    nd = 0
    for i in range(0,Nelec):
      for j in range(i+1,Nelec):
        for a in range(Nelec,dim):
          for b in range(a+1,dim):
            nd += 1
    #printall = False
    print 'No of unique double-excited determinants: ', nd
    print 'Problem size: %dx%d' % (nd+1,nd+1)
    
    print 'DCI matrix equations:'
    print '( 0  B ) (1) = Ecorr*(1)'
    print '( B+ D ) (c) =       (c)'
    # Construction of the B block of normal-ordered Hamiltonian matrix
    t0 = time.time()
    B = np.zeros(nd)
    iajb = -1
    for i in range(0,Nelec):
      for j in range(i+1,Nelec):
        for a in range(Nelec,dim):
          for b in range(a+1,dim):
            iajb += 1
            B[iajb] = ints[i,j,a,b]
            
    #print 'B = ', B
    del B
    
    
    





