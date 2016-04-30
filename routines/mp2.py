#!/usr/bin/python

from __future__ import division
import math
import numpy as np

#####################################################
#
#  MP2 ENERGY CALCULATION
#
#####################################################
def mp2_energy(dim,Nelec,ints,E):
    CC = 0.0
    for i in range(0,Nelec):
	for j in range(0,Nelec):
	    for a in range(Nelec,dim*2):
		for b in range(Nelec,dim*2):
		    CC += 0.25*(ints[i,j,a,b]*ints[i,j,a,b])/(E[i//2] + E[j//2] - E[a//2] - E[b//2])

    return CC

##################################################
#
#  THIRD-ORDER CORRECTION TO ENERGY (MP3)
#
##################################################
def mp3_energy(dim,Nelec,ints,E):
    diag1 = 0.0
    diag2 = 0.0
    diag3 = 0.0
    for i in range(0,Nelec):
        for j in range(0, Nelec):
            for k in range(0, Nelec):
                for a in range(Nelec,dim*2):
                    for b in range(Nelec,dim*2):
                        for c in range(Nelec,dim*2):
                            diag1 -= ints[i,k,a,c]*ints[a,j,b,k]*ints[b,c,i,j]/(E[i//2]+E[k//2]-E[a//2]-E[c//2])*(E[i//2]+E[j//2]-E[b//2]-E[c//2])
    for i in range(0,Nelec):
        for j in range(0, Nelec):
            for a in range(Nelec,dim*2):
                for b in range(Nelec,dim*2):
                    for c in range(Nelec,dim*2):
                        for d in range(Nelec,dim*2):
                            diag2 += 0.125*ints[i,j,a,c]*ints[a,c,b,d]*ints[b,d,i,j]/(E[i//2]+E[j//2]-E[a//2]-E[c//2])*(E[i//2]+E[j//2]-E[b//2]-E[d//2])
    for i in range(0,Nelec):
        for j in range(0, Nelec):
            for k in range(0, Nelec):
                for l in range(0, Nelec):
                    for a in range(Nelec,dim*2):
                        for b in range(Nelec,dim*2):
                            diag3 += 0.125*ints[i,j,a,b]*ints[k,l,i,j]*ints[a,b,k,l]/(E[i//2]+E[j//2]-E[a//2]-E[b//2])*(E[k//2]+E[l//2]-E[a//2]-E[b//2])
    return diag1 + diag2 + diag3



