import numpy as np
from functools import lru_cache
from scipy.integrate import quad,nquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmaz, brmesolve
hbar=kb=1
#Def spectraldensity 
def J(w, w0, lam, Td, gamma0,omega_c, spectrum_form):
    if spectrum_form=="drude-norm":
        return (gamma0 * lam**2) / (2*np.pi * ((w - w0)**2 + lam**2)) * (np.cos(w*Td/2)**2)
    if spectrum_form=="expo": 
        return gamma0*np.exp(-(w - w0) / (omega_c)) * (np.cos(w*Td/2)**2)
    if spectrum_form=="drude-under-damped":
        return (gamma0 * lam**2) / (2*np.pi * ((omega_c**2 - w0**2) + lam**2*w**2)) * (np.cos(w*Td/2)**2)
    else:
        return "kein spektrum angegeben"




def kernel_maker(T, w0, lam, Td, gamma0, omega_c, spectrum_form,
                 epsabs=1e-10, epsrel=1e-8, limit=400, wmin=1e-6):

    beta = 1.0 / T

    def coth(x):
        # stabil bei x->0
        if abs(x) < 1e-6:
            return 1.0 / x
        return 1.0 / np.tanh(x)

    pool = ThreadPoolExecutor(max_workers=2)

    @lru_cache(maxsize=200000)
    #rückwert und vorwarts zwei punkte corr; 
    def f(tau):
        tau = float(tau)
        if tau < 0:
            return np.conjugate(f(-tau))
        # C(tau) = ∫ J(w)[coth(beta w/2) cos(w tau) - i sin(w tau)] dw
        def re_int(w):
            return (J(w, w0, lam, Td, gamma0, omega_c, spectrum_form) * coth(0.5 * beta * w) * np.cos(w * tau))

        def im_int(w):
            return -J(w, w0, lam, Td, gamma0, omega_c, spectrum_form) * np.sin(w * tau)
        # beide quad-Aufrufe parallel starten
        fut_re = pool.submit(quad, re_int, wmin, omega_c, epsabs=epsabs, epsrel=epsrel, limit=limit)
        fut_im = pool.submit(quad, im_int, wmin, omega_c, epsabs=epsabs, epsrel=epsrel, limit=limit)

        Cr, _ = fut_re.result()
        Ci, _ = fut_im.result()
        return Cr + 1j * Ci

    return f
#Tcl zweiter ordnung: 
def Kernel2(f, epsabs=1e-8, epsrel=1e-8):
    pool = ThreadPoolExecutor(max_workers=2)
    def B(t):
        t = float(t)
        if t < 0:
            raise ValueError("t must be >= 0")

        def integrand(t1):
            return f(t - t1)

        def integrand_re(t1):
            return float(np.real(integrand(t1)))

        def integrand_im(t1):
            return float(np.imag(integrand(t1)))
        fut_re_B=pool.submit(quad, integrand_re, 0.0, t, epsabs=epsabs, epsrel=epsrel,limit=100)
        fut_im_B= pool.submit(quad, integrand_im, 0.0, t, epsabs=epsabs, epsrel=epsrel,limit=100)
        Bre, _ = fut_re_B.result()
        Bim, _ = fut_im_B.result()
        return Bre + 1j * Bim

    return B




#TCl vierter ordnung: 

def Kernel4(f, epsabs=1e-6, epsrel=1e-6):
    pool = ThreadPoolExecutor(max_workers=2)
    def A(t):
        t = float(t)
        if t < 0:
            raise ValueError("t must be >= 0")

        def integrand(t3, t2, t1):
            return f(t - t2) * f(t1 - t3) + f(t - t3) * f(t1 - t2)

        def integrand_re(t3, t2, t1):
            return float(np.real(integrand(t3, t2, t1)))

        def integrand_im(t3, t2, t1):
            return float(np.imag(integrand(t3, t2, t1)))

        bounds = [
            lambda t2, t1: [0.0, t2],   # t3
            lambda t1:     [0.0, t1],   # t2
            [0.0, t]                    # t1
        ]
        Are, _ = nquad(integrand_re, bounds, opts={"epsabs": epsabs, "epsrel": epsrel})
        Aim, _ = nquad(integrand_im, bounds, opts={"epsabs": epsabs, "epsrel": epsrel})

        return Are + 1j * Aim
    return A

def solver(y, x):
    I = np.zeros_like(y, dtype=complex)
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        I[i] = I[i-1] + 0.5 * dx * (y[i] + y[i-1])
    return I


    



if __name__ == "__main__":
    import qutip as qutip
    from qutip import basis, sigmax, sigmaz, brmesolve

    # -------- Badparameter (dein Stil) --------
    w0      = 3.0

    omega_c = 10.0
    gamma0  = 0.1

    lam = 1
    Td = 0.0
 
    T = 0.1
    alpha=1

    omega_c = 10.0 * w0
    spectrum_form = "expo"
    rho11_0 = 1.0
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    f = kernel_maker(T=T, w0=w0, lam=lam, Td=Td, gamma0=gamma0,
                 omega_c=omega_c, spectrum_form=spectrum_form, wmin=1e-6)

    B = Kernel2(f)
    A = Kernel4(f)   # falls du A wirklich willst; sonst auskommentieren

    tlist = np.linspace(0.0, 10.0, 50)

    def compute_B_vals():
        return np.array([B(float(t)) for t in tlist], dtype=complex)

    def compute_A_vals():
        return np.array([A(float(t)) for t in tlist], dtype=complex)

    with ThreadPoolExecutor(max_workers=2) as pool2:
        futB = pool2.submit(compute_B_vals)
        futA = pool2.submit(compute_A_vals)   # wenn A aus ist: diese Zeile raus

        B_vals = futB.result()
        A_vals = futA.result()                # wenn A aus ist: diese Zeile raus



    

    # kernen übger tlist berechnen.
    #B_vals = np.array([B(float(t)) for t in tlist], dtype=complex)
    #A_vals = np.array([A(float(t)) for t in t_grid], dtype=complex)

    # time-dependent decay rate for rho11:
    # rho11(t) = rho11(0) * exp( - ∫_0^t 2 Re[ B(s) + A(s) ] ds )
    rate = 2.0 * np.real(alpha**2 * B_vals + alpha**4 * A_vals)

    # optional: prevent unphysical growth if instantaneous rate becomes negative
    rate = np.maximum(rate, 0.0)

    Irate = solver(rate.astype(complex), tlist)  # integral of rate
    rho11 = rho11_0 * np.exp(-np.real(Irate))


    

    # Kopplungsoperator (z.B. σx für Relaxation zwischen Eigenzuständen)
    P00 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    P11 = qutip.basis(2, 1) * qutip.basis(2, 1).dag()
    P01 = qutip.basis(2, 0) * qutip.basis(2, 1).dag()   # sigma_-
    P10 = qutip.basis(2, 1) * qutip.basis(2, 0).dag()   # sigma_+
    sigma_x = P01 + P10
    
    # Anfangszustand: excited state |1>
    rho0 = basis(2, 1) * basis(2, 1).dag()

    def nbar(w, T):
        if T <= 0:
            return 0.0
        x = w / T
        if x < 1e-8:
            return 1.0 / x
        return 1.0 / (np.exp(x) - 1.0)



# QuTiP-kompatibel: S(omega, args)
    def S_not_markov_hot(w):
        wabs = max(abs(w), 1e-6 )

    # exponentieller Cutoff, bei w0 ~ O(1) in etwa 1 und ansonsten <=1
        cutoff = np.exp(-(wabs - w0) / (omega_c))
        cutoff = min(cutoff, 1.0)  # verhindert Verstärkung für w < w0

        if w > 0:
            return 2*np.pi*gamma0 * (nbar(wabs, T) + 1.0) * cutoff
        elif w < 0:
            return 2*np.pi*gamma0 * nbar(wabs, T) * cutoff
        else:
            return 0.0   # absorption


# --- setup ---


    rho0 = basis(2, 1) * basis(2, 1).dag()
    P1   = basis(2, 1) * basis(2, 1).dag()
    H    = w0 * P1



    args = {"w0": w0, "Td": Td, "omega_c": omega_c, "gamma0": gamma0, "T": T}

# Kopplungsoperator
    

# Debug: prüfen, ob QuTiP wirklich nicht-null Spektrum sieht


    res_BR = qutip.brmesolve(H, rho0, tlist,a_ops=[[sigma_x,S_not_markov_hot]],e_ops=[P11])

    rho11_red = np.array(res_BR.expect[0], dtype=float)







    # plots
    plt.figure(figsize=(7, 4))
    plt.plot(tlist, np.real(B_vals), label="Re B(t)")
    #plt.plot(t_grid, np.real(A_vals), label="Re A(t)")
    plt.xlabel("t")
    plt.ylabel("kernels")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tlist, rate, label="2 Re(B+A)")
    plt.xlabel("t")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tlist, rho11, label="rho11(t)")
    plt.plot(tlist, rho11_red,label="red",color="red")
    plt.xlabel("t")
    plt.ylabel(r"$\rho_{11}(t)$")
    plt.xlabel("t")
    plt.ylabel("rho11")
    plt.legend()
    plt.tight_layout()
    plt.savefig("TCL.png", dpi=300, bbox_inches="tight")
    plt.show()

