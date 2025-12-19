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
        cutoff = np.exp(-(w - w0)/omega_c)
        cutoff = min(cutoff, 1.0)   # wichtig!
        return gamma0 * cutoff * (np.cos(w*Td/2)**2)  
    if spectrum_form=="ohmic":
        return gamma0*w*np.exp(-w/omega_c)  * (np.cos(w*Td/2)**2)
    
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
        fut_re = pool.submit(quad, re_int, wmin, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
        fut_im = pool.submit(quad, im_int, wmin, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)

        Cr, _ = fut_re.result()
        Ci, _ = fut_im.result()
        return Cr + 1j * Ci

    return f
#Tcl zweiter ordnung: 
from scipy.interpolate import interp1d




def Kernel2(f, w0, sign=+1, epsabs=1e-8, epsrel=1e-8):
    # sign=+1 -> e^{+i w0 tau}  (down)
    # sign=-1 -> e^{-i w0 tau}  (up)
    pool = ThreadPoolExecutor(max_workers=2)

    def B(t):
        t = float(t)
        if t < 0:
            raise ValueError("t must be >= 0")

        def integrand(tau):
            return f(tau) * np.exp(1j * sign * w0 * tau)

        def integrand_re(tau): return float(np.real(integrand(tau)))
        def integrand_im(tau): return float(np.imag(integrand(tau)))

        fut_re = pool.submit(quad, integrand_re, 0.0, t, epsabs=epsabs, epsrel=epsrel, limit=400)
        fut_im = pool.submit(quad, integrand_im, 0.0, t, epsabs=epsabs, epsrel=epsrel, limit=400)
        Bre, _ = fut_re.result()
        Bim, _ = fut_im.result()
        return Bre + 1j * Bim

    return B




#TCl vierter ordnung: 

from scipy.integrate import nquad
import numpy as np

def Kernel4_pm(f_plus, f_minus, epsabs=1e-6, epsrel=1e-6):
    """
    Gibt vier getrennte TCL4-Kerne zurück:
    App, Apm, Amp, Amm
    sodass A = App + Apm + Amp + Amm (für lineares Bad, Wick/Gauss).
    """

    def _A_of(fa, fb):
        def A(t):
            t = float(t)
            if t < 0:
                raise ValueError("t must be >= 0")

            def integrand(t3, t2, t1):
                #  f -> fa/fb getrennt
                return fa(t - t2) * fb(t1 - t3) + fa(t - t3) * fb(t1 - t2)

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

    App = _A_of(f_plus,  f_plus)
    Apm = _A_of(f_plus,  f_minus)
    Amp = _A_of(f_minus, f_plus)
    Amm = _A_of(f_minus, f_minus)

    return App, Apm, Amp, Amm
"""
def solver(y, x):
    I = np.zeros_like(y, dtype=complex)
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        I[i] = I[i-1] + 0.5 * dx * (y[i] + y[i-1])
    return I

def solve_rho11_population(tlist, Gdown, Gup, rho11_0=1.0):
    rho = np.zeros_like(tlist, dtype=float)
    rho[0] = float(rho11_0)

    for k in range(1, len(tlist)):
        dt = tlist[k] - tlist[k-1]
        gd = float(Gdown[k-1])
        gu = float(Gup[k-1])

        rho[k] = rho[k-1] + dt * (-gd * rho[k-1] + gu * (1.0 - rho[k-1]))

        # numerical safety
        if rho[k] < 0.0: rho[k] = 0.0
        if rho[k] > 1.0: rho[k] = 1.0

    return rho
    
"""




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from qutip import basis, sigmax, sigmaz, brmesolve
    # params
    w0 = 1.0
    omega_c = 10.0 * w0
    gamma0 = 0.1
    lam = 1.0
    Td = 1
    T = 0.5
    alpha = 1.0
    spectrum_form = "ohmic"
    rho11_0 = 1.0
    tlist = np.linspace(0.0, 10.0, 100)
    f = kernel_maker(T=T, w0=w0, lam=lam, Td=Td, gamma0=gamma0,
                     omega_c=omega_c, spectrum_form=spectrum_form, wmin=1e-6)

    # build B±
    # vorher
    B_plus  = Kernel2(f, w0=w0, sign=+1)
    B_minus = Kernel2(f, w0=w0, sign=-1)



    # build f± (needed for TCL4 split)
    def f_plus(tau):
        tau = float(tau)
        if tau < 0:
            return np.conjugate(f_plus(-tau))
        return f(tau) * np.exp(1j * w0 * tau)

    def f_minus(tau):
        tau = float(tau)
        if tau < 0:
            return np.conjugate(f_minus(-tau))
        return f(tau) * np.exp(-1j * w0 * tau)

    # build TCL4 kernels
    #App, Apm, Amp, Amm = Kernel4_pm(f_plus, f_minus)

    

    # compute B± and A-terms
    def compute_Bp(): return np.array([B_plus(float(t))  for t in tlist], dtype=complex)
    def compute_Bm(): return np.array([B_minus(float(t)) for t in tlist], dtype=complex)

    # TCL4 is expensive: compute only if you really want it
    def compute_App(): return np.array([App(float(t)) for t in tlist], dtype=complex)
    def compute_Apm(): return np.array([Apm(float(t)) for t in tlist], dtype=complex)
    def compute_Amp(): return np.array([Amp(float(t)) for t in tlist], dtype=complex)
    def compute_Amm(): return np.array([Amm(float(t)) for t in tlist], dtype=complex)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futBp  = pool.submit(compute_Bp)
        futBm  = pool.submit(compute_Bm)

        #futApp = pool.submit(compute_App)
        #futApm = pool.submit(compute_Apm)
        #futAmp = pool.submit(compute_Amp)
        #futAmm = pool.submit(compute_Amm)

        Bp_vals  = futBp.result()
        Bm_vals  = futBm.result()
        #App_vals = futApp.result()
        #Apm_vals = futApm.result()
        #Amp_vals = futAmp.result()
        #Amm_vals = futAmm.result()

    # mapping to "down/up" for TCL4 (minimal consistent choice):
    #A_down = App_vals + Apm_vals   # fa = f_plus
    #A_up   = Amm_vals + Amp_vals   # fa = f_minus

    # rates including TCL2 + TCL4
    Gdown = 2.0 * np.real(alpha**2 * Bp_vals )#+ alpha**4 * A_down)
    Gup   = 2.0 * np.real(alpha**2 * Bm_vals )#+ alpha**4 * A_up)
    from scipy.interpolate import PchipInterpolator
    gd = PchipInterpolator(tlist, Gdown, extrapolate=True)
    gu = PchipInterpolator(tlist, Gup,   extrapolate=True)

    # avoid negative instantaneous rates (numerical safeguard)
    #Gdown = np.maximum(Gdown, 0.0)
    #Gup   = np.maximum(Gup, 0.0)

    # thermalising rho11(t)
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    #gd = interp1d(tlist, Gdown, kind="cubic", fill_value="extrapolate")
    #gu = interp1d(tlist, Gup,   kind="cubic", fill_value="extrapolate")

    def rhs(t, y):
        r = y[0]
        return [-gd(t)*r + gu(t)*(1.0 - r)]

    sol = solve_ivp(rhs, (tlist[0], tlist[-1]), [rho11_0], t_eval=tlist, rtol=1e-8, atol=1e-10)
    rho11 = sol.y[0]


    #rho11 = solve_rho11_population(tlist, Gdown, Gup, rho11_0=rho11_0)



    import qutip as qutip

    # Kopplungsoperator (z.B. σx für Relaxation zwischen Eigenzuständen)
    P00 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    P11 = qutip.basis(2, 1) * qutip.basis(2, 1).dag()
    P01 = qutip.basis(2, 0) * qutip.basis(2, 1).dag()   # sigma_-
    P10 = qutip.basis(2, 1) * qutip.basis(2, 0).dag()   # sigma_+
    sigma_x = P01 + P10
    H    = w0 * P11
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
        wabs = max(abs(w), 1e-12)
        Jw = gamma0 * wabs * np.exp(-wabs / omega_c) * (np.cos(w*Td/2)**2) # ohmic exp cutoff, >=0

        nb = nbar(wabs, T)
        if w > 0:
            return float(2*np.pi * Jw * (nb + 1.0))    # emission
        elif w < 0:
            return float(2*np.pi * Jw * nb)            # absorption
        else:
            return 0.0

    



    args = {"w0": w0, "Td": Td, "omega_c": omega_c, "gamma0": gamma0, "T": T}


    res_BR = qutip.brmesolve(H, rho0, tlist,a_ops=[[sigma_x,S_not_markov_hot]],e_ops=[P11])

    rho11_red = np.array(res_BR.expect[0], dtype=float)







    # plots
    plt.figure(figsize=(7, 4))
    plt.plot(tlist, np.real(Bp_vals), label="Re B_plus(t)")
    plt.plot(tlist, np.real(Bm_vals), label="Re B_minus(t)")
    plt.xlabel("t"); plt.ylabel("B kernels")
    plt.legend(); plt.tight_layout(); plt.show()

    #plt.figure(figsize=(7, 4))
    #plt.plot(tlist, np.real(A_down), label="Re A_down(t)")
    #plt.plot(tlist, np.real(A_up),   label="Re A_up(t)")
    #plt.xlabel("t"); plt.ylabel("A kernels")
    #plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tlist, Gdown, label="Gamma_down(t)")
    plt.plot(tlist, Gup,   label="Gamma_up(t)")
    plt.xlabel("t"); plt.ylabel("rates")
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tlist, rho11, label="TCL2+TCL4 rho11(t)")
    plt.plot(tlist, rho11_red, label="BR rho11(t)", color="red")
    plt.xlabel("t"); plt.ylabel(r"$\rho_{11}(t)$")
    plt.legend(); plt.tight_layout()
    plt.savefig("TCL.png", dpi=300, bbox_inches="tight")
    plt.show()


