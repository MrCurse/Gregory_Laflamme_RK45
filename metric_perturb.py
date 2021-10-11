'''
Runge-Kutta code for solving the metric perturbation equation for general
D-dimensions
=====================================================================
Differential equations

H_tr'  = G_tr
G_tr' = P(r)G_tr + Q(r)H_tr

=====================================================================
'''
import numpy as np
from numpy import sqrt, exp
from numba import jit
import time
import matplotlib.pyplot as plt
#import multiprocessing as mp


@jit(nopython=True)
def V(r):
    return 1-2/r

'''
Define the functions P(r) and Q(r) here, which also depend on constant parameters
mu and Omega for the numerical scan
'''

@jit(nopython=True)
def P(r, mu, Omega, D, rp=2):
    num = mu*mu*((D-2)-2*(rp/r)**(D-3)+(4-D)*(rp/r)**(2*D-6))/(r*V(r))+Omega*Omega*((D-2)+\
            (2*D-7)*(rp/r)**(D-3))/(r*V(r))-( 3*(D-3)**2 *(rp/r)**(2*D-6) *((D-2)-\
                (rp/r)**(D-3) ) )/(4*V(r)*r**3)

    denom = -Omega**2 - mu**2 *V(r) + ((D-3)**2 *(rp/r)**(2*D-6))/(4*r**2)

    return num/denom

@jit(nopython=True)
def Q(r, mu, Omega, D, rp=2):
    num = (mu**2 +Omega**2 /V(r))**2 +(Omega**2 /(4*r*r*V(r)**2))*(4*D-8-(8*D-16)*\
            (rp/r)**(D-3) -(53-34*D+5*D**2)*(rp/r)**(2*D-6))+(mu**2 /(4*r*r*V(r)))*\
                (4*D-8-4*(3*D-7)*(rp/r)**(D-3)+(D**2 +2*D-11)*(rp/r)**(2*D-6))+\
                    ((D-3)**2 /(4*r**4 *V(r)**2))*(rp/r)**(2*D-6) *((D-2)*(2*D-5)-
                         (D-1)*(D-2)*(rp/r)**(D-3)+(rp/r)**(2*D-6))

    denom = -Omega**2 - mu**2 *V(r) + ((D-3)**2 *(rp/r)**(2*D-6))/(4*r**2)

    return -num/denom


'''
Defines the stepper for Runge-Kutta-Fehlberg method, which can be used to define
both 4th and 5th order RK for use in adaptive step-size
'''
@jit(nopython=True)
def RK45_solve(h, mu, Omega, D, rtol):
    C30 = 3./8
    C31 = 3./32
    C32 = 9./32
    C40 = 12./13
    C41 = 1932./2197
    C42 = -7200./2197
    C43 = 7296./2197
    C51 =  439./216
    C52 = -8
    C53 = 3680./513
    C54 = -845./4104
    C61 = -8.27
    C62 = 2
    C63 = -3544./2565
    C64 = 1859./4104
    C65 = -11./40

    CW1 = 25./216
    CW3 = 1408./2565
    CW4 = 2197./4104
    CW5 = -1./5

    CZ1 = 16./135
    CZ3 = 6656./12825
    CZ4 = 28561./56430
    CZ5 = -9./50
    CZ6 = 2./55

    CE1 = 1./360
    CE3 = -128./4275
    CE4 = -2197./75240
    CE5 = 1./50
    CE6 = 2./55

    atol = 10**-12
    alpha = 0.8
    k = 0
    r1 = 200.0
    r2 = 2.001
    r = np.empty(1); Htr = np.empty(1); Gtr = np.empty(1)
    htr0 = exp(-sqrt(mu**2 + Omega**2)*r1); gtr0 = -sqrt(mu**2 + Omega**2)*htr0
    r[0] = r1
    Htr[0] = htr0
    Gtr[0] = gtr0

    ri = r1; Hi = htr0; Gi = gtr0 # defined for th purpose of iterating in a while loop

    while ri > r2:

        K1 = Gi
        L1 = P(ri,mu,Omega,D)*Gi+Q(ri,mu,Omega,D)*Hi
        K2 = Gi+0.25*h*L1
        L2 = P(ri+0.25*h,mu,Omega,D)*(Gi+0.25*h*L1)+Q(ri+0.25*h,mu,Omega,D)*(Hi+0.25*h*K1)
        K3 = Gi+C31*h*L1+C32*h*L2
        L3 = P(ri+C30*h,mu,Omega,D)*(Gi+C31*h*L1+C32*h*L2)+Q(ri+C30*h,mu,Omega,D)*\
            (Hi+C31*h*K1+C32*h*K2)
        K4 = Gi+C41*h*L1+C42*h*L2+C43*h*L3
        L4 = P(ri+C40*h,mu,Omega,D)*(Gi+C41*h*L1+C42*h*L2+C43*h*L3)+Q(ri+C40*h,mu,Omega,D)*\
            (Hi+C41*h*K1+C42*h*K2+C43*h*K3)
        K5 = Gi+C51*h*L1+C52*h*L2+C53*h*L3+C54*h*L4
        L5 = P(ri+h,mu,Omega,D)*(Gi+C51*h*L1+C52*h*L2+C53*h*L3+C54*h*L4)+Q(ri+h,mu,Omega,D)*\
            (Hi+C51*h*K1+C52*h*K2+C53*h*K3+C54*h*K4)
        K6 = Gi+C61*h*L1+C62*h*L2+C63*h*L3+C64*h*L4+C65*h*L5
        L6 = P(ri+0.5*h,mu,Omega,D)*(Gi+C61*h*L1+C62*h*L2+C63*h*L3+C64*h*L4+C65*h*L5)+\
            Q(ri+0.5*h,mu,Omega,D)*(Hi+C61*h*K1+C62*h*K2+C63*h*K3+C64*h*K4+C65*h*K5)

        HSol_4 = Hi + h*(CW1*K1 + CW3*K3 + CW4*K4 + CW5*K5)
        GSol_4 = Gi + h*(CW1*L1 + CW3*L3 + CW4*L4 + CW5*L5)

        HSol_5 = Hi + h*(CZ1*K1 + CZ3*K3 + CZ4*K4 + CZ5*K5 + CZ6*K6)
        GSol_5 = Hi + h*(CZ1*L1 + CZ3*L3 + CZ4*L4 + CZ5*L5 + CZ6*L6)

        err = h*np.absolute(CE1*K1 + CE3*K3 + CE4*K4 + CE5*K5 + CE6*K6)
        #tol = rtol*np.absolute(Hi) + atol
        tol = rtol

        if err <= tol:
            ri = ri - h
            h = alpha*h*(tol/err)**0.2
            r = np.append(r, ri)
            Hi = HSol_5
            Gi = GSol_5
            Htr = np.append(Htr, Hi)
            Gtr = np.append(Gtr, Gi)
            k = 0
        elif k == 0:
            h = alpha*h*(tol/err)**0.2
            k = k + 1
        else:
            h = 0.5*h

    return r, Htr, Gtr

if __name__ == "__main__":

    D = 4
    rp = 2
    dr = 0.1
    Omega = 0.04; mu = 0.2
    r, Htr, Gtr = RK45_solve(dr, mu, Omega, D, 10**-8)

    r = np.flip(r)
    Htr = np.flip(Htr)
    Gtr = np.flip(Gtr)

    plt.plot(r, (r-rp)*Htr)
    plt.show()
