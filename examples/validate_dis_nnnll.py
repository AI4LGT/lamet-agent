"""Historical DIS Figs. 5/6 benchmark for the shared hard evolution.

Run: python examples/validate_dis_nnnll.py
Reference: hep-ph/0607228v1 (50)-(55), (63), (93)-(97).
This benchmark uses fixed nf=5, historical Pade Gamma3, and the inferred
four-loop inverse-log coupling alpha_s(MZ)=0.1187. Production H_F instead
uses fixed nf=3, the exact cusp and numerical running from the supplied
alpha_s(mu). No curve fitting is performed by this script.
"""
import numpy as np
from scipy.special import digamma, polygamma, gammaln
from lamet_agent.kernels.tmd.cg_nnnll import _evolution_orders

BETA = np.array([7.666666666666666, 38.666666666666664, 180.9074074074073, 4826.156328790895])
CUSP = np.array([5.333333333333333, 36.84359134233823, 239.20803319895958, 1553.0])
GV = np.array([-8.0, 1.1419392681384153, -249.387567105445])
GJ = np.array([-4.0, 38.67634642520811, 204.8155493094977])
J1 = np.array([0.5603516434761258, -4.0, 2.6666666666666665, 0.0, 0.0])
J2 = np.array([-36.34110881644504, 32.138910584653274, 43.249400053772106, -17.48148148148148, 3.5555555555555554])
C1, C2 = -8.473421244202363, -48.606058107065394

# Vertices extracted from the published PDF, separately calibrated per panel.
REFERENCE = [[5, 5, 0, 0.7957093779655994, 1.704096572743301],
 [5, 5, 0, 0.9473617448396267, 4.868598918337467],
 [5, 5, 1, 0.7957093779655994, 1.8912421191023998],
 [5, 5, 1, 0.9299956682181427, 4.5616936086293975],
 [5, 5, 2, 0.7957093779655994, 1.8592105656318187],
 [5, 5, 2, 0.9299956682181427, 4.627190964233425],
 [5, 30, 0, 0.7973073475547013, 1.3905220067529211],
 [5, 30, 0, 0.9492628865246845, 2.5229569426599348],
 [5, 30, 1, 0.7973073475547013, 1.4693339707771844],
 [5, 30, 1, 0.9492628865246845, 2.7776556010398306],
 [5, 30, 2, 0.7973073475547013, 1.4541309349508489],
 [5, 30, 2, 0.9492628865246845, 2.7706994950249517],
 [6, 5, 0, 0.7957080158635791, 0.7385052944779907],
 [6, 5, 0, 0.9520856986824895, 1.4188238382777634],
 [6, 5, 1, 0.7957080158635791, 0.8209766342872153],
 [6, 5, 1, 0.9520856986824895, 1.7881158659315664],
 [6, 5, 2, 0.7957080158635791, 0.8141419445173714],
 [6, 5, 2, 0.9520856986824895, 1.868836965082699],
 [6, 30, 0, 0.7973062555900458, 0.9029539739805398],
 [6, 30, 0, 0.9492634292629031, 1.282540723734558],
 [6, 30, 1, 0.7973062555900458, 0.9548442112167921],
 [6, 30, 1, 0.9492634292629031, 1.4078102109981416],
 [6, 30, 2, 0.7973062555900458, 0.9465529681862904],
 [6, 30, 2, 0.9492634292629031, 1.4063408767902046]]


def alpha(mu):
    b = BETA
    L = np.log((mu/.21664629271351624)**2)
    log_L = np.log(L)
    a = 1/(b[0]*L)-b[1]*log_L/(b[0]**3*L**2)
    a += (b[1]**2*(log_L*log_L-log_L-1)+b[0]*b[2])/(b[0]**5*L**3)
    a += (b[1]**3*(-log_L**3+2.5*log_L*log_L+2*log_L-.5)-3*b[0]*b[1]*b[2]*log_L+b[0]**2*b[3]/2)/(b[0]**7*L**4)
    return a  # alpha_s/(4*pi)


def a_orders(anom, a0, a1):
    b = BETA/BETA[0]
    c = np.convolve(anom, [1., -b[1], b[1]**2-b[2]])[:3]
    return c*np.array([np.log(a1/a0), a1-a0, (a1*a1-a0*a0)/2])/(2*BETA[0])


def moments(eta):
    d1 = -np.euler_gamma-digamma(5+eta)
    d2, d3, d4 = [-polygamma(k, 5+eta) for k in [1, 2, 3]]
    return np.array([1, d1, d1*d1+d2, d1**3+3*d1*d2+d3,
                     d1**4+6*d1*d1*d2+3*d2*d2+4*d1*d3+d4])


def dis_k(x, Q, muf, order):
    mi = Q*np.sqrt((1-x)/x)
    ah, ai, af = [alpha(u) for u in (Q, mi, muf)]
    eta = 2*a_orders(CUSP, ai, af)
    # Use production hard evolution; jet/PDF evolution is benchmark-only.
    E = 2*_evolution_orders(ah, ai, BETA, CUSP, GV)
    E += 2*a_orders(GJ-GV, ai, af)+eta*(np.log((1-x)/x)-np.euler_gamma)
    base = np.exp(E[0]+gammaln(5)-gammaln(5+eta[0]))
    if order == 0:
        return base
    e1 = E[1]-digamma(5+eta[0])*eta[1]
    e2 = E[2]-digamma(5+eta[0])*eta[2]-polygamma(1, 5+eta[0])*eta[1]**2/2
    m = moments(eta[0])
    m1 = 2*C1*ah+ai*(J1@m)
    if order == 1:
        return base*(1+e1+m1)
    m2 = (C1*C1+2*C2)*ah*ah+ai*ai*(J2@m)+2*C1*ah*ai*(J1@m)
    dm = np.array([0, -polygamma(1, 5+eta[0]),
                   -2*m[1]*polygamma(1, 5+eta[0])-polygamma(2, 5+eta[0]), 0, 0])
    return base*(1+e1+m1+e2+e1*e1/2+m2+e1*m1+eta[1]*ai*(J1@dm))


def benchmark_errors():
    rows = []
    for figure, Q, order, x, expected in REFERENCE:
        muf = Q if figure == 5 else {5: 1.5, 30: 10}[Q]
        actual = dis_k(x, Q, muf, order)
        rows.append((figure, Q, order, x, 100*(actual/expected-1)))
    return rows


if __name__ == "__main__":
    errors = benchmark_errors()
    print("Maximum relative deviation from 24 published vertices [%]:",
          max(abs(r[-1]) for r in errors))
    if max(abs(r[-1]) for r in errors) > .005:
        raise SystemExit("Historical DIS benchmark failed")
