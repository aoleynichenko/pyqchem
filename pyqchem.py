#!/usr/bin/python
from __future__ import division
import sys
import os
import numpy as np
#import matplotlib.pyplot as plt
import math
import pyximport; pyximport.install()
from routines.integral_input import __init_integrals__
import routines.scf as scf
import routines.ints as ints
import routines.detci.detci as detci
import routines.ao2mo as ao2mo
import routines.mp2 as mp2
import routines.cistdhf as cistdhf
import routines.ccsd as ccsd
import routines.eomccsd as eomccsd
import routines.eommbpt2 as eommbpt2
import routines.eommbptp2 as eommbptp2
import routines.eommbptd as eommbptd
import time

""" Edit below to perform the calculation desired
"""

do_DIIS      = True
do_ao2mo     = False
do_mp2       = False
do_mp3       = False
do_cis       = False
do_dci       = False
do_cistdhf   = False
do_ccsd      = False
do_eomccsd   = False
do_eommbpt2  = False
do_eommbptp2 = False
do_eommbptd  = False
printops     = True
reuse_t1t2   = False
convergence  = 1.0e-8
printnum     = 6			# number eigenvalues to print for TDHF/CIS

""" Here the main routine runs
"""

# ask for location of integral/input files

if len(sys.argv)==1:
    inp_name = 'input'
else:
    inp_name = sys.argv[1]


#LOCATION = sys.argv[1]
PYQCHEM_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))

print
print "\t\t*******************************************"
print "\t\t*              P Y Q C H E M              *"
print "\t\t*              =============              *"
print "\t\t*                                         *"
print "\t\t* Experimental quantum chemistry software *"
print "\t\t*                                         *"
print "\t\t* Based on the pyqchem code written by    *"
print "\t\t* Joshua Goings:                          *"
print "\t\t* http://joshuagoings.com                 *"
print "\t\t* https://github.com/jjgoings/pyqchem     *"
print "\t\t*                                         *"
print "\t\t* Author:                                 *"
print "\t\t* Alexander Oleynichenko                  *"
print "\t\t* Lomonosov Moscow State University       *"
print "\t\t* alexvoleynichenko@gmail.com             *"
print "\t\t*******************************************"
print

print "\n\t*** Begin quantum chemistry on:   " + inp_name

t1 = time.time()

# read input file
def read_input(file):
    global do_ao2mo, do_mp2, do_ccsd, do_eomccsd, do_cis, reuse_t1t2

    inp = {}
    execfile(file, inp)

    print '---------------------- echo of input file ----------------------'
    os.system('cat ' + file)
    print '---------------------- end of input file -----------------------'

    print '\n\tInput file data:'
    print '\t----------------'
    try:
        # task
        task = inp['task']
        print '\ttask = ', task
        if task == 'scf':
            pass
        elif task == 'ao2mo':
            do_ao2mo = True
        elif task == 'ccsd':
            do_ao2mo = True
            do_ccsd = True
        elif task == 'eomccsd':
            do_ao2mo = True
            do_ccsd = True
            do_eomccsd = True
        elif task == 'cis':
            do_ao2mo = True
            do_cis = True
        elif task == 'mp2':
            do_ao2mo = True
            do_mp2 = True
        else:
            raise Exception('wrong task ' + task)

        # geom
        geom = inp['geom']
        # bohrs to angstroms
        for i,a in enumerate(geom):
            factor = 1.88971616463207
            geom[i][1] *= factor  # x
            geom[i][2] *= factor  # y
            geom[i][3] *= factor  # z
        print '\tgeom = ', geom
        # get unique elements in molecule
        uniq_elems = set([a[0] for a in geom])

        # basis set
        basis = inp['basis']

        if isinstance(basis, basestring):
            basis_lib_location = PYQCHEM_HOME + '/basis_library'
            bas_file = basis_lib_location + '/' + basis
            bas_inp = {}
            execfile(bas_file, bas_inp)
            basis = bas_inp['basis']

        print '\t basis = \n'
        for k,v in basis.items():
            if k in uniq_elems:
                print 'Element: ', k
                for fun in v:
                    L, e, c = fun
                    print 'L = ', L
                    for i,ei in enumerate(e):
                        print '%12.6f%12.6f' % (e[i], c[i])
                print
        print '\t----------------\n\n'

        # reuse cluster amplitudes or not
        try:
            reuse_t1t2 = bool(inp['reuse_t'])
        except KeyError:
            pass

        # molecular charge
        try:
            charge = int(inp['charge'])
        except KeyError:
            charge = 0
    except KeyError as e:
        print 'Section not found!'
        print e
        raise

    return geom, basis, charge

geom, basis, charge = read_input(inp_name)

# create one and two electron integral arrays, as well as determine
# number of basis functions (dim) and number of electrons (Nelec)

print "\n\t*** Started AO integrals evaluation"

bfns = ints.calculate_ints(geom, basis, charge)  # returns atom-centered basis set
LOCATION = '.'
ENUC,Nelec,dim,S,T,V,Hcore,twoe = __init_integrals__(LOCATION)

print "\t*** Finished AO integrals evaluation\n"

# do SCF iteration
print "\t*** Begin SCF, convergence requested: ",convergence,"a.u."
EN,orbitalE,C,P,F = scf.scf_iteration(convergence,ENUC,Nelec,dim,S,Hcore,twoe,printops,do_DIIS,geom,bfns)

print "Total E(RHF): ", EN+ENUC," a.u."
print "\t*** End of SCF\n"

if do_ao2mo == True:
    ints,fs = ao2mo.transform_ao2mo(dim,twoe,C,orbitalE)


if do_mp2 == True:
    # Begin MP2 calculation
    mp2_corr = mp2.mp2_energy(dim,Nelec,ints,orbitalE)
    print "Ecorr(MP2)   = ",mp2_corr," a.u."
    print "Total E(MP2) = ",mp2_corr+EN+ENUC," a.u."
    if do_mp3 == True:
        mp3_corr = mp2.mp3_energy(dim,Nelec,ints,orbitalE)
        print "Ecorr(MP3)   = ",mp3_corr," a.u."
        print "Total E(MP3) = ",mp2_corr+mp3_corr+EN+ENUC," a.u.\n"

if do_cis == True:
    detci.cis(EN,fs,ints,Nelec,dim)
if do_dci == True:
    detci.cid(EN,fs,ints,Nelec,dim)

if do_cistdhf == True:
    print "\n\t*** Begin CIS/TDHF calculation"
    cistdhf.cistdhf(Nelec,dim,fs,ints,printnum)
    print "\t*** End CIS/TDHF calculation"

if do_ccsd == True:
    print "\n\t*** Begin CCSD calculation"
    ECCSD,T1,T2 = ccsd.ccsd(Nelec,dim,fs,ints,convergence,printops,reuse_t1t2)
    print "E(CCSD): {0:.8f}".format(ECCSD+ENUC+EN),"a.u."
    print "\t*** End CCSD calculation"

elif ao2mo == True and do_ccsd == False:
    # bad amplitudes for EOM if CCSD is off
    T1 = np.zeros((dim*2,dim*2))
    T2 = np.zeros((dim*2,dim*2,dim*2,dim*2))
    for a in range(Nelec,dim*2):
        for b in range(Nelec,dim*2):
            for i in range(0,Nelec):
                for j in range(0,Nelec):
                    T2[a,b,i,j] += ints[i,j,a,b]/(fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

if do_eomccsd == True:
    print "\n\t*** Begin EOM-CCSD calculation"
    eomccsd.eomccsd(Nelec,dim,fs,ints,T1,T2)
    print "\t*** End EOM-CCSD calculation"

if do_eommbpt2 == True:
    print "\n\t*** Begin EOM-MBPT2 calculation"
    eommbpt2.eommbpt2(Nelec,dim,fs,ints,T1,T2)
    print "\t*** End EOM-MBPT2 calculation"

if do_eommbptd == True:
    print "\n\t*** Begin EOM-MBPT(D) calculation"
    eommbptd.eommbptd(Nelec,dim,fs,ints,T1,T2)
    print "\t*** End EOM-MBPT(D) calculation"

if do_eommbptp2 == True:
    print "\n\t*** Begin EOM-MBPT(2) calculation"
    eommbptp2.eommbptp2(Nelec,dim,fs,ints,T1,T2)
    print "\t*** End EOM-MBPT(2) calculation"

# end of pyqchem
t2 = time.time()
print "*** Total time (sec): %.2f" % (t2-t1)

