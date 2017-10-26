# AO integrals evaluation module
#
# Based on the excellent code of Joshua Goings: http://joshuagoings.com/
# McMurchie-Davidson algorithm is implemented.
# Alexander Oleynichenko, 2017

cimport cython
cimport numpy as np

import math
import numpy as np
import time
from scipy.special import hyp1f1


cdef double PI = 3.1415926535897932384626433832795


cdef int fact2(int n):
    return reduce(int.__mul__, range(n, 0, -2), 1)


cdef double boys(int n, double T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)


class BasisFunction(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  tuple of angular momentum
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps  = np.array(exps)
        self.coefs = np.array(coefs)
        self.norm = None
        self.normalize()

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
            do not integrate to unity.
        '''
        l,m,n = self.shell
        # self.norm is a list of length equal to number primitives
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                        np.power(self.exps,l+m+n+1.5)/
                        fact2(2*l-1)/fact2(2*m-1)/
                        fact2(2*n-1)/np.power(PI,1.5))


cdef double E(int i, int j, int t, double Qx, double a, double b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral,
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    cdef double p, q

    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)



def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    cdef int l1, m1, n1, l2, m2, n2
    cdef double S1, S2, S3

    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b)#,n[0],A[0]) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b)#,n[1],A[1]) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b)#,n[2],A[2]) # Z
    return S1*S2*S3*np.power(PI/(a+b),1.5)


def int_S(a,b):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s


def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    cdef int l1, m1, n1, l2, m2, n2
    cdef double term0, term1, term2

    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2


def int_T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t


cdef double R(int t, int u, int v, int n, double p, double PCx, double PCy, double PCz, double RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function
        PCx,y,z: Cartesian vector distance between Gaussian
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    cdef double T, val

    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val


def gaussian_product_center(a, A, b, B):
    return (a*A+b*B)/(a+b)


def nuclear_attraction(a, lmn1, A, b, lmn2, B, C):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
     '''
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    p = a + b
    P = gaussian_product_center(a, A, b, B)  # Gaussian composite center
    RPC = np.linalg.norm(P - C)

    val = 0.0
    for t in xrange(l1 + l2 + 1):
        for u in xrange(m1 + m2 + 1):
            for v in xrange(n1 + n2 + 1):
                val += E(l1, l2, t, A[0] - B[0], a, b) * \
                       E(m1, m2, u, A[1] - B[1], a, b) * \
                       E(n1, n2, v, A[2] - B[2], a, b) * \
                       R(t, u, v, 0, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], RPC)
    val *= 2 * PI / p
    return val


def int_V(a,b,C):
    '''Evaluates overlap between two contracted Gaussians
        Returns float.
        Arguments:
        a: contracted Gaussian 'a', BasisFunction object
        b: contracted Gaussian 'b', BasisFunction object
        C: center of nucleus
     '''
    v = 0.0
    for ia, ca in enumerate(a.coefs):
         for ib, cb in enumerate(b.coefs):
             v += a.norm[ia]*b.norm[ib]*ca*cb*\
                      nuclear_attraction(a.exps[ia],a.shell,a.origin,
                      b.exps[ib],b.shell,b.origin,C)
    return v


def electron_repulsion(a, lmn1, A, b, lmn2, B, c, lmn3, C, d, lmn4, D):
    ''' Evaluates electron repulsion integral between four primitive Gaussians
         Returns a float.
         a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
         lmn1,lmn2
         lmn3,lmn4: int tuple containing orbital angular momentum
                    for Gaussian 'a','b','c','d', respectively
         A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
     '''
    cdef int l1, m1, n1, l2, m2, n2, l3, m3, n3, l4, m4, n4
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    l3, m3, n3 = lmn3
    l4, m4, n4 = lmn4
    cdef double p = a + b  # composite exponent for P (from Gaussians 'a' and 'b')
    cdef double q = c + d  # composite exponent for Q (from Gaussians 'c' and 'd')
    cdef double alpha = p * q / (p + q)
    P = gaussian_product_center(a, A, b, B)  # A and B composite center
    Q = gaussian_product_center(c, C, d, D)  # C and D composite center
    cdef double RPQ = np.linalg.norm(P - Q)

    # distances
    cdef double A0B0 = A[0] - B[0]
    cdef double A1B1 = A[1] - B[1]
    cdef double A2B2 = A[2] - B[2]
    cdef double C0D0 = C[0] - D[0]
    cdef double C1D1 = C[1] - D[1]
    cdef double C2D2 = C[2] - D[2]
    cdef double P0Q0 = P[0] - Q[0]
    cdef double P1Q1 = P[1] - Q[1]
    cdef double P2Q2 = P[2] - Q[2]
    cdef int t, u, v, tau, nu, phi

    cdef double val = 0.0
    for t in xrange(l1 + l2 + 1):
        for u in xrange(m1 + m2 + 1):
            for v in xrange(n1 + n2 + 1):
                for tau in xrange(l3 + l4 + 1):
                    for nu in xrange(m3 + m4 + 1):
                        for phi in xrange(n3 + n4 + 1):
                            val += E(l1, l2, t, A0B0, a, b) * \
                                   E(m1, m2, u, A1B1, a, b) * \
                                   E(n1, n2, v, A2B2, a, b) * \
                                   E(l3, l4, tau, C0D0, c, d) * \
                                   E(m3, m4, nu,  C1D1, c, d) * \
                                   E(n3, n4, phi, C2D2, c, d) * \
                                   np.power(-1, tau + nu + phi) * \
                                   R(t + tau, u + nu, v + phi, 0, \
                                     alpha, P0Q0, P1Q1, P2Q2, RPQ)

    val *= 2 * np.power(PI, 2.5) / (p * q * np.sqrt(p + q))
    return val


def ERI(a,b,c,d):
     '''Evaluates overlap between two contracted Gaussians
        Returns float.
        Arguments:
        a: contracted Gaussian 'a', BasisFunction object
        b: contracted Gaussian 'b', BasisFunction object
        c: contracted Gaussian 'c', BasisFunction object
        d: contracted Gaussian 'd', BasisFunction object
     '''
     cdef double eri = 0.0
     cdef int ja, jb, jc, jd
     cdef double ca, cb, cc, cd

     for ja, ca in enumerate(a.coefs):
         for jb, cb in enumerate(b.coefs):
             for jc, cc in enumerate(c.coefs):
                 for jd, cd in enumerate(d.coefs):
                     eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                              ca*cb*cc*cd*\
                              electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                 b.exps[jb],b.shell,b.origin,\
                                                 c.exps[jc],c.shell,c.origin,\
                                                 d.exps[jd],d.shell,d.origin)
     return eri


def gen_angmom_list(L):
    shells = []
    for i in xrange(0, L+1):
        for j in xrange(0, L+1):
            for k in xrange(0, L+1):
                if i + j + k == L:
                    shells.append((i,j,k))
    return shells


# returns atom-centered basis set
def calculate_ints(geom, basis, charge):

    # atom-centered basis set
    bfns = []
    for atom in geom:
        Z = atom[0]
        x, y, z = [float(coord) for coord in atom[1:]]
        # print Z, x, y, z

        for b in basis[Z]:
            # gen shells
            for sh in gen_angmom_list(b[0]):
                bfns.append(BasisFunction(origin=(x,y,z), shell=sh, exps=b[1], coefs=b[2]))

    M = len(bfns)
    print 'AO basis dim = ', M

    S, T, V = np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M))
    for i, a in enumerate(bfns):
        for j, b in enumerate(bfns):
            S[i, j] = int_S(a, b)
            T[i, j] = int_T(a, b)
            for at in geom:
                V[i, j] += at[0] * int_V(a, b, at[1:])

    enuc = 0.0
    for a in geom:
        for b in geom:
            d = math.sqrt((a[1]-b[1])**2+(a[2]-b[2])**2+(a[3]-b[3])**2)
            if d > 0:  # exclude self-interaction
                enuc += a[0]*b[0] / d
    enuc /= 2

    nelec = -charge
    for a in geom:
        nelec += a[0]

    print 'nelec = ', nelec
    print 'enuc  = ', enuc

    # only unique integrals will be evaluated
    print '# unique ERIs  = ',  (M**4+2*M**3+3*M**2+2*M)/8
    n_nonzero = 0
    count = 0
    t1 = time.time()
    with open("eri.dat", "w") as f:
        for m in xrange(0, M):
            for n in xrange(m, M):
                for p in xrange(m, M):
                    q0 = n if p == m else p
                    for q in xrange(q0,M):
                        eri_mnpq = ERI(bfns[m], bfns[n], bfns[p], bfns[q])
                        count += 1
                        if count % 5000 == 0:
                            print 'done: ', count
                        if abs(eri_mnpq) > 1e-12:
                            n_nonzero += 1
                            f.write("%3d%3d%3d%3d%15.8f\n" % (m+1, n+1, p+1, q+1, eri_mnpq))
    t2 = time.time()

    print '# nonzero ERIs = ', n_nonzero
    print '%% nonzero ERIs = %.1f' % (100.0*n_nonzero/count)
    print 'Time for 2-el integrals = ', (t2-t1), ' sec'
    print 'Time per ERI = ', (t2-t1)/count, ' sec'

    # print evaluated integrals to files
    np.savetxt('s.dat', S)
    np.savetxt('t.dat', T)
    np.savetxt('v.dat', V)
    with open("enuc.dat", "w") as f:
        f.write("%.8f\n" % (enuc))
    with open("nbf.dat", "w") as f:
        f.write("%d\n" % (M))
    with open("nelec.dat", "w") as f:
        f.write("%d\n" % (nelec))

    return bfns
