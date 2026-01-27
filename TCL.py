
__all__ = ['TCLsolve_TCL2', 'TCL_solve_time_dep', 'gamma_maker_with_J',
           'build_L_TCL2_time_dependent', 'make_G_from_Gamma']


import numpy as np
import scipy.integrate
import scipy.sparse as sp
import numpy as np
from functools import lru_cache
from scipy.integrate import quad,nquad
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmaz, brmesolve
from concurrent.futures import ThreadPoolExecutor
from qutip import Qobj, isket, liouvillian, expect
from qutip import spre, spost
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.integrate import quad
import numpy as np
from scipy.integrate import quad
import numpy as np
# Options + _solver_safety_check: in QuTiP 5 evtl. anders exportiert
try:
    from qutip.solver import Options, _solver_safety_check
except ImportError:
    from qutip.solver import Options
    def _solver_safety_check(*args, **kwargs):
        return
    

class TCLResult:
    def __init__(self, solver, times, expect=None, states=None):
        self.solver = solver
        self.times = np.array(times)
        self.expect = expect if expect is not None else []
        self.states = states if states is not None else []


hbar=kb=1
#Def spectraldensity 
def J(w, w0, lam, Td, gamma0, omega_c, spectrum_form):
    if spectrum_form == "drude-norm":
        return (gamma0 * lam**2) / (2*np.pi * ((w - w0)**2 + lam**2)) * (np.cos(w*Td/2)**2)

    if spectrum_form == "ohmic":
        return gamma0*w*np.exp(-w/omega_c) * (np.cos(w*Td/2)**2)

    if spectrum_form == "drude-under-damped":
        return (gamma0 * lam**2) / (2*np.pi * ((omega_c**2 - w0**2) + lam**2*w**2)) * (np.cos(w*Td/2)**2)

    raise ValueError(f"Unknown spectrum_form='{spectrum_form}'. Use e.g. 'ohmic', 'drude-norm', 'drude-under-damped'.")

        




# -------------------------------------------------------------------------
# eigene kleine Helfer statt alter qutip.cy.*-Funktionen
# -------------------------------------------------------------------------

def mat2vec(M):
    """
    Matrix -> Vektor (Spaltenvektorisierung, Fortran-Order),
    kompatibel zum alten QuTiP-4-Verhalten.
    """
    M = np.asarray(M)
    return np.reshape(M, (M.size,), order='F')


def vec2mat(v, shape):
    """
    Vektor -> Matrix (Inverse von mat2vec).
    shape = (N, N) oder allgemein (nrow, ncol).
    """
    v = np.asarray(v)
    return np.reshape(v, shape, order='F')


def vec2mat_index(N, I):
    """
    I -> (a, b) bei Spaltenvektorisierung.
    """
    a = I % N
    b = I // N
    return a, b


def TCLsolve_TCL(
    H, psi0, tlist, a_ops,
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    e_ops,
    c_ops=None,
    options=None,
    use_secular=False, dw_min=0.0,
    sign_convention="plus",
    use_TCL2=True,
    use_TCL4=False,
    TCL4_builder=None,
):
    if e_ops is None:
        e_ops = []
    if c_ops is None:
        c_ops = []
    if options is None:
        options = Options()

    # 1) Basis-Liouvillian
    L0 = liouvillian(H, c_ops)

    # 2) Eigenbasis
    evals, ekets = H.eigenstates()
    N = len(evals)
    W = np.subtract.outer(evals, evals)

    K = len(a_ops)
    A = np.zeros((K, N, N), dtype=complex)
    for r, Ar in enumerate(a_ops):
        A[r] = Ar.transform(ekets).full()

    # 3) Liouville-Indizes
    Iabs = np.array([(a + N*b, a, b)
                     for a in range(N) for b in range(N)],
                     dtype=np.int32)

    # 4) TCL2-Kern (optional)
    if use_TCL2:
        C, Gamma = gamma_maker_with_J(
            T=T, w0=w0, lam=lam, Td=Td,
            gamma0=gamma0, omega_c=omega_c,
            spectrum_form=spectrum_form,
            sign_convention=sign_convention,
        )
        G, Gc = make_G_from_Gamma(Gamma, W)

        L2_of_t = build_L_TCL2_time_dependent(
            L0=L0, A=A, W=W, Iabs=Iabs,
            K=K, N=N, G=G, Gc=Gc,
            use_secular=use_secular,
            dw_min=dw_min
        )
    else:
        L2_of_t = lambda t, args=None: L0

    # 5) TCL4-Kern (optional)
    if use_TCL4:
        if TCL4_builder is None:
            L4_of_t = L4_of_t = build_L_TCL4_time_dependent(L0=L0, A=A, W=W, Iabs=Iabs, K=K, N=N,T=T, w0=w0, lam=lam, Td=Td, gamma0=gamma0, omega_c=omega_c,
            spectrum_form=spectrum_form,
            tmax=float(np.max(tlist)),
            use_secular=use_secular, dw_min=dw_min,
            sign_convention=sign_convention,
            ntau=2001)


    
    else:
        L4_of_t = lambda t, args=None: 0 * L0

    # 6) Gesamter Generator
    def L_of_t(t, args=None):
        return L2_of_t(t) + L4_of_t(t)

    # 7) Zeitintegration
    results = TCL_solve_time_dep(
        L_of_t, ekets, psi0, tlist,
        e_ops=e_ops, options=options
    )

    tag = "tcl2+tcl4" if use_TCL4 else "tcl2"
    if e_ops:
        return TCLResult(tag, tlist, expect=results)
    else:
        return TCLResult(tag, tlist, states=results)







import scipy.integrate
import numpy as np
from qutip import Qobj, isket, expect

def TCL_solve_time_dep(L_of_t, ekets, rho0, tlist, e_ops, options=None):
    """
    TCL2: löst dρ/dt = L(t) ρ mit L_of_t(t) -> Qobj (Superoperator in Eigenbasis).
    """
    if e_ops is None:
        e_ops = []
    if options is None:
        options = Options()

    # Initial state
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    n_tsteps = len(tlist)
    result_list = []

    # in Eigenbasis
    rho_eb = rho0.transform(ekets)
    e_eb_ops = [e.transform(ekets) for e in e_ops]

    for e_eb in e_eb_ops:
        #result_list.append(np.zeros(n_tsteps, dtype=float if e_eb.isherm else complex))
        result_list.append(np.zeros(n_tsteps, dtype=complex))
    template = rho_eb
    y0 = mat2vec(rho_eb.full())

    # RHS: dy/dt = L(t) y
    # Wichtig: L_of_t(t) liefert Qobj, daraus Matrix holen
    #from qutip.core.data import matmul

    def rhs(t, y):
        L_t = L_of_t(t)              # Qobj
        Lsp = L_t.data.as_scipy()     # -> scipy.sparse.csr_matrix
        return Lsp.dot(y)

    # sparse-dot (schneller als full()), wenn CSR/CSC

    r = scipy.integrate.ode(rhs)
    r.set_integrator('zvode')
    r.set_initial_value(y0, tlist[0])

    store_states = len(e_ops) == 0
    states_out = [] if store_states else None

    for t_idx, t in enumerate(tlist):
        if not r.successful():
            break

        rho_mat = vec2mat(r.y, template.shape)
        rho_eb = Qobj(rho_mat, dims=template.dims)

        if e_ops:
            for m, e in enumerate(e_eb_ops):
                result_list[m][t_idx] = expect(e, rho_eb)
        else:
            states_out.append(rho_eb.transform(ekets, True))

        if t_idx < n_tsteps - 1:
            r.integrate(tlist[t_idx + 1])

    return result_list if e_ops else states_out






def gamma_maker_with_J(
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    # numerik für C(tau):
    wmin=1e-3, wmax=100.0, nw=10000,
    # numerik für Γ(t):
    epsabs=1e-10, epsrel=1e-8, limit=1200,
    # Vorzeichenkonvention für Γ:
    sign_convention="plus",  # "plus": exp(+i Ω τ), "minus": exp(-i Ω τ)
):
    """
    Liefert zwei Funktionen:
      C(tau):  komplexe Badkorrelation aus J(ω)
      Gamma(t, Omega): TCL2-Kern Γ(t,Ω)=∫_0^t dτ C(τ) exp(± i Ω τ)

    Parameter:
      T: Temperatur (k_B=1)
      spectrum_form: String, der die Spektraldichte auswählt
      (w0, lam, Td, gamma0, omega_c): Parameter, die in J(ω) eingehen (je nach Form)
    """

    beta = 1.0 / float(T)


    def coth(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        small = np.abs(x) < 1e-6
        out[small] = 1.0 / x[small]
        out[~small] = 1.0 / np.tanh(x[~small])
        return out


    wgrid = np.linspace(float(wmin), float(wmax), int(nw))
    Jvals = J(wgrid, w0, lam, Td, gamma0, omega_c, spectrum_form)
    def C(tau):
        tau = float(tau)

        # parallel Re/Im bauen (wie bei dir)
        def re_vals():
            return Jvals * coth(0.5 * beta * wgrid) * np.cos(wgrid * tau)

        def im_vals():
            return -Jvals * np.sin(wgrid * tau)

        
        yre = Jvals * coth(0.5 * beta * wgrid) * np.cos(wgrid * tau)
        yim = -Jvals * np.sin(wgrid * tau)
        Cr = np.trapezoid(yre, wgrid)
        Ci = np.trapezoid(yim, wgrid)
        return Cr + 1j*Ci


    # ========= TCL2 Kernel Γ(t,Ω) =========
    pool = ThreadPoolExecutor(max_workers=2)

    def phase(Omega, tau):
        if sign_convention == "plus":
            return np.exp(1j * Omega * tau)
        elif sign_convention == "minus":
            return np.exp(-1j * Omega * tau)
        else:
            raise ValueError("sign_convention must be 'plus' or 'minus'")

    def Gamma(t, Omega):
        t = float(t)
        if t < 0:
            raise ValueError("t must be >= 0")
        Omega = float(Omega)

        def integrand(tau):
            return C(tau) * phase(Omega, tau)

        def integrand_re(tau): return float(np.real(integrand(tau)))
        def integrand_im(tau): return float(np.imag(integrand(tau)))
        """
        fut_re = pool.submit(quad, integrand_re, 0.0, t,
                             epsabs=epsabs, epsrel=epsrel, limit=limit)
        fut_im = pool.submit(quad, integrand_im, 0.0, t,
                             epsabs=epsabs, epsrel=epsrel, limit=limit)

        Gre, _ = fut_re.result()
        Gim, _ = fut_im.result()
        return Gre + 1j * Gim"""
        Gre, _ = quad(integrand_re, 0.0, t, epsabs=epsabs, epsrel=epsrel, limit=limit)
        Gim, _ = quad(integrand_im, 0.0, t, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return Gre + 1j*Gim


    return C, Gamma


def make_G_from_Gamma(Gamma, W, t_round=8, cache_size=50_000):
    """
    Erzeugt G und Gc mit LRU-Cache für Gamma(t, Ω).

    Parameter
    ---------
    Gamma : callable
        Funktion Gamma(t, Omega)
    W : ndarray
        Bohrfrequenzen W[i,j]
    t_round : int
        Dezimalstellen zum Runden der Zeit (wichtig!)
    cache_size : int
        Größe des LRU-Caches
    """

    @lru_cache(maxsize=cache_size)
    def Gamma_cached(t_rounded, Omega):
        return Gamma(t_rounded, Omega)

    def G(r, i, j, t):
        t_key = round(float(t), t_round)
        return Gamma_cached(t_key, float(W[i, j]))

    def Gc(r, i, j, t):
        # exakt das, was in der Formel steht: komplex konjugiert
        t_key = round(float(t), t_round)
        return np.conjugate(Gamma_cached(t_key, float(W[i, j])))

    return G, Gc

def build_L_TCL2_time_dependent(L0, A, W, Iabs, K, N,G,Gc, use_secular=False,dw_min=0.0):
    """
    Baut einen zeitabhängigen TCL2-Generator L(t) = L0 + R_TCL2(t)
    analog zum Bloch-Redfield-Tensor-Builder, aber mit zeitabhängigen Kernen G.

    Parameter
    ---------
    L0 : Qobj
        Basis-Liouvillian (z.B. Hamiltonteil + evtl. andere Dissipation)
    A : np.ndarray
        Systemoperatoren in Eigenbasis: A[r, i, j]
    W : np.ndarray
        Bohrfrequenzen: W[i, j] = E_i - E_j (oder deine Konvention)
    Iabs : np.ndarray
        Liste der relevanten (Index, a, b) für die Liouville-Indizierung
        wie in deinem Code.
    K : int
        Anzahl Kopplungsoperatoren
    N : int
        Systemdimension
    G : callable
        TCL2-Kernfunktion: G(r, i, j, t)
    use_secular : bool
        wie gehabt
    dw_min : float
        wie gehabt

    Rückgabe
    --------
    L_of_t : callable
        Funktion L_of_t(t, args=None) -> Qobj (zeitabhängiger Superoperator)
    """

    # zur Laufzeit schnellere Sicht auf Iabs-Spalten
    Iabs = np.asarray(Iabs)
    I_idx = Iabs[:, 0].astype(np.int32)
    I_a   = Iabs[:, 1].astype(np.int32)
    I_b   = Iabs[:, 2].astype(np.int32)

    # wir bauen eine Map von (a,b) -> I (Superindex), damit wir schnell rows/cols füllen können
    # (optional, aber praktisch)
    # Falls Iabs schon alle Paare enthält, geht das.
    idx_map = {(int(a), int(b)): int(I) for I, a, b in Iabs}

    dims = L0.dims

    
    def build_L_t(t):
        rows = []
        cols = []
        data = []

        # optional: secular clustering je (a,b) separat
        for I, a, b in Iabs:
            if use_secular:
                # finde alle (c,d) mit ähnlicher Frequenz
                mask = np.abs(W[a, b] - W[I_a, I_b]) < (dw_min / 10.0)
                Jcds = Iabs[mask]
            else:
                Jcds = Iabs

            for J, c, d in Jcds:
                elem = 0.0 + 0.0j

                # 1) direkter Term: 1/2 * sum_r A_ac A_db [G_ca(t) + G_db(t)]
                for r in range(K):
                    elem += 0.5*2 * (
                        A[r, a, c] * A[r, d, b] *
                        (G(r, c, a, t) + Gc(r, d, b, t))
                    )

                # 2) erster Spur-Term: -1/2 * delta_{b d} * sum_{r,k} A_ak A_kc G_ck(t)
                if b == d:
                    for r in range(K):
                        for k in range(N):
                            elem -= 0.5*2 * (
                                A[r, a, k] * A[r, k, c] * G(r, c, k, t)
                            )

                # 3) zweiter Spur-Term: -1/2 * delta_{a c} * sum_{r,k} A_dk A_kb G_dk(t)
                if a == c:
                    for r in range(K):
                        for k in range(N):
                            elem -= 0.5*2 * (
                                A[r, d, k] * A[r, k, b] * Gc(r, d, k, t)
                            )

                if elem != 0.0:
                    rows.append(int(I))
                    cols.append(int(J))
                    data.append(elem)

        R_sparse = sp.coo_matrix(
            (np.array(data, dtype=complex),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(N**2, N**2),
            dtype=complex
        ).tocsr()

        R_qobj = Qobj(R_sparse, dims=dims)
        return L0 + R_qobj

    # QuTiP-kompatible Signatur (t, args)
    def L_of_t(t, args=None):
        return build_L_t(t)

    return L_of_t
"""
def TCL_C4(
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    # numerik für C(tau):
    wmin=1e-3, wmax=100.0, nw=10000,
    # numerik für Γ(t):
    epsabs=1e-10, epsrel=1e-8, limit=1200,
    # Vorzeichenkonvention für Γ:
    sign_convention="plus",  # "plus": exp(+i Ω τ), "minus": exp(-i Ω τ)
):
    

    beta = 1.0 / float(T)


    def coth(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        small = np.abs(x) < 1e-6
        out[small] = 1.0 / x[small]
        out[~small] = 1.0 / np.tanh(x[~small])
        return out


    wgrid = np.linspace(float(wmin), float(wmax), int(nw))
    Jvals = J(wgrid, w0, lam, Td, gamma0, omega_c, spectrum_form)
    def C(tau):
        tau = float(tau)

        # parallel Re/Im bauen (wie bei dir)
        def re_vals():
            return Jvals * coth(0.5 * beta * wgrid) * np.cos(wgrid * tau)

        def im_vals():
            return -Jvals * np.sin(wgrid * tau)

        
        yre = Jvals * coth(0.5 * beta * wgrid) * np.cos(wgrid * tau)
        yim = -Jvals * np.sin(wgrid * tau)
        Cr = np.trapezoid(yre, wgrid)
        Ci = np.trapezoid(yim, wgrid)
        return Cr + 1j*Ci
    

    def phase(Omega, tau):
        if sign_convention == "plus":
            return np.exp(1j * Omega * tau)
        elif sign_convention == "minus":
            return np.exp(-1j * Omega * tau)
        else:
            raise ValueError("sign_convention must be 'plus' or 'minus'") 
        
    
    return C,phase


"""

import numpy as np

def TCL_C4(
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    wmin=1e-3, wmax=100.0, nw=10000,
    sign_convention="plus",
    # --- NEU: τ-Tabellierung ---
    tmax=10.0,      # max tau, den du brauchst (>= max t0 in TCL4)
    ntau=2001,      # Anzahl τ-Stützstellen (größer = genauer, langsamer Precompute)
):
    beta = 1.0 / float(T)

    def coth(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        small = np.abs(x) < 1e-6
        out[small] = 1.0 / x[small]
        out[~small] = 1.0 / np.tanh(x[~small])
        return out

    # Frequenzgitter + Faktoren einmal
    wgrid = np.linspace(float(wmin), float(wmax), int(nw))
    Jvals = J(wgrid, w0, lam, Td, gamma0, omega_c, spectrum_form)
    cothfac = coth(0.5 * beta * wgrid)

    # ---------- PRECOMPUTE C(tau) auf einem τ-Gitter ----------
    taus = np.linspace(0.0, float(tmax), int(ntau))
    Ctab = np.empty_like(taus, dtype=complex)

    # Achtung: das ist der eine "teure" Schritt, aber nur einmal!
    for n, tau in enumerate(taus):
        yre = Jvals * cothfac * np.cos(wgrid * tau)
        yim = -Jvals * np.sin(wgrid * tau)
        Ctab[n] = np.trapezoid(yre, wgrid) + 1j * np.trapezoid(yim, wgrid)

    dtau = taus[1] - taus[0]

    # ---------- schnelle lineare Interpolation ----------
    def C(tau):
        tau = float(tau)
        if tau <= 0.0:
            return Ctab[0]
        if tau >= tmax:
            # entweder clampen oder Fehler werfen – clamp ist oft ok
            return Ctab[-1]

        x = tau / dtau
        i = int(x)
        a = x - i
        return (1.0 - a) * Ctab[i] + a * Ctab[i + 1]

    def phase(Omega, tau):
        Omega = float(Omega); tau = float(tau)
        if sign_convention == "plus":
            return np.exp(1j * Omega * tau)
        elif sign_convention == "minus":
            return np.exp(-1j * Omega * tau)
        else:
            raise ValueError("sign_convention must be 'plus' or 'minus'")

    return C, phase


def Convolute_4_3_pairs(C, phase, epsabs=1e-10, epsrel=1e-8, limit=400):
    """
    K(t0) = ∫_0^{t0} dt1 ∫_0^{t1} dt2 ∫_0^{t2} dt3
            Ca(t_i - t_j) e^{iΩa(t_i-t_j)} * Cb(t_k - t_l) e^{iΩb(t_k-t_l)}

    Du wählst die Argumente über Indexpaare:
      pairA = (i,j)  bedeutet tauA = t_i - t_j
      pairB = (k,l)  bedeutet tauB = t_k - t_l
    mit t_0=t0, t_1=t1, t_2=t2, t_3=t3.
    """

    def _pick_part(which):
        if which == "full": return lambda tau: C(tau)
        if which == "re":   return lambda tau: np.real(C(tau))
        if which == "im":   return lambda tau: np.imag(C(tau))
        raise ValueError("part must be 'full','re','im'")

    def Falten(Omega_a, Omega_b, pairA, pairB, part_a="full", part_b="full"):
        Omega_a = float(Omega_a); Omega_b = float(Omega_b)
        i, j = map(int, pairA)
        k, l = map(int, pairB)
        Ca = _pick_part(part_a)
        Cb = _pick_part(part_b)

        def K(t0):
            t0 = float(t0)

            def int_t1(t1):
                def int_t2(t2):
                    def int_t3(t3):
                        times = (t0, t1, t2, t3)
                        tauA = times[i] - times[j]
                        tauB = times[k] - times[l]
                        return (Ca(tauA) * phase(Omega_a, tauA) *
                                Cb(tauB) * phase(Omega_b, tauB))
                    return quad(int_t3, 0.0, t2, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
                return quad(int_t2, 0.0, t1, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
            return quad(int_t1, 0.0, t0, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

        return K

    return Falten

from functools import lru_cache




def C4(falten_pairs, W):
    """
    Liefert Cf mit expliziter Wahl von re/im für beide Kerne
    """

    def Cf(i, j, k, l, a, c, b, d, partA="full", partB="full"):
        Omega_a = float(W[a, c])
        Omega_b = float(W[b, d])
        return falten_pairs(
            Omega_a, Omega_b,
            pairA=(i, j),
            pairB=(k, l),
            part_a=partA,
            part_b=partB,
        )

    return Cf

class K4Pairs:
    def __init__(self, W, falten_pairs, t_round=6):
        self.W = W
        self.falten = falten_pairs
        self.t_round = t_round
        self._cache = {}

    @staticmethod
    def _part(p):
        p = p.lower()
        if p in ("c", "full"): return "full"
        if p in ("r", "re"):   return "re"
        if p in ("i", "im"):   return "im"
        raise ValueError("part muss c/r/i (oder full/re/im) sein.")

    def get(self, ij, ab, partA="c", partB="c", pairA=(0,2), pairB=(1,3)):
        """
        ij=(i,j) wählt Ω_ij, ab=(a,b) wählt Ω_ab.
        pairA=(u,v) wählt tauA = t_u - t_v (u,v in {0,1,2,3})
        pairB=(u,v) wählt tauB = t_u - t_v
        """
        i,j = ij
        a,b = ab
        key = (i,j,partA,a,b,partB,pairA,pairB)
        if key not in self._cache:
            Kfun = self.falten(
                self.W[i,j], self.W[a,b],
                pairA=pairA, pairB=pairB,
                part_a=self._part(partA),
                part_b=self._part(partB),
            )
            @lru_cache(200_000)
            def f(t0):  # TCL4 liefert Funktion von t0
                return Kfun(round(float(t0), self.t_round))
            self._cache[key] = f
        return self._cache[key]
    


        
def build_L_TCL4_time_dependent(
    L0, A, W, Iabs, K, N,
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    tmax,
    use_secular=False, dw_min=0.0, sign_convention="plus",
    ntau=2001
):

    ...
    
    """
    Baut einen zeitabhängigen TCL2-Generator L(t) = L0 + R_TCL2(t)
    analog zum Bloch-Redfield-Tensor-Builder, aber mit zeitabhängigen Kernen G.

    Parameter
    ---------
    L0 : Qobj
        Basis-Liouvillian (z.B. Hamiltonteil + evtl. andere Dissipation)
    A : np.ndarray
        Systemoperatoren in Eigenbasis: A[r, i, j]
    W : np.ndarray
        Bohrfrequenzen: W[i, j] = E_i - E_j (oder deine Konvention)
    Iabs : np.ndarray
        Liste der relevanten (Index, a, b) für die Liouville-Indizierung
        wie in deinem Code.
    K : int
        Anzahl Kopplungsoperatoren
    N : int
        Systemdimension
    G : callable
        TCL2-Kernfunktion: G(r, i, j, t)
    use_secular : bool
        wie gehabt
    dw_min : float
        wie gehabt

    Rückgabe
    --------
    L_of_t : callable
        Funktion L_of_t(t, args=None) -> Qobj (zeitabhängiger Superoperator)
    """

    # zur Laufzeit schnellere Sicht auf Iabs-Spalten
    Iabs = np.asarray(Iabs)
    I_idx = Iabs[:, 0].astype(np.int32)
    I_a   = Iabs[:, 1].astype(np.int32)
    I_b   = Iabs[:, 2].astype(np.int32)


    # wir bauen eine Map von (a,b) -> I (Superindex), damit wir schnell rows/cols füllen können
    # (optional, aber praktisch)
    # Falls Iabs schon alle Paare enthält, geht das.
    idx_map = {(int(a), int(b)): int(I) for I, a, b in Iabs}

    dims = L0.dims
    """
    Cbath, phase = TCL_C4(
    T=T, w0=w0, lam=lam, Td=Td,
    gamma0=gamma0, omega_c=omega_c,
    spectrum_form=spectrum_form,
    sign_convention=sign_convention,
    )"""
    Cbath, phase = TCL_C4(
    T=T, w0=w0, lam=lam, Td=Td, gamma0=gamma0, omega_c=omega_c,
    spectrum_form=spectrum_form,
    wmin=1e-3, wmax=100.0, nw=10000,
    sign_convention=sign_convention,
    tmax=tmax,
    ntau=ntau
    )


    falten_pairs = Convolute_4_3_pairs(Cbath, phase)
    K4 = K4Pairs(W, falten_pairs, t_round=4)   # 4 ist ein guter Start

    #Cf = C4(falten_pairs, W)



# z.B. imaginär × reell
    
    from functools import lru_cache


    def build_L_t(t):
        rows = []
        cols = []
        data = []

        # optional: secular clustering je (a,b) separat
        
        for I, a, b in Iabs:
            if use_secular:
                # finde alle (c,d) mit ähnlicher Frequenz
                mask = np.abs(W[a, b] - W[I_a, I_b]) < (dw_min / 10.0)
                Jcds = Iabs[mask]
            else:
                Jcds = Iabs

            for J, c, d in Jcds:
                elem = 0.0 + 0.0j

                # 1) direkter Term: 1/2 * sum_r A_ac A_db [G_ca(t) + G_db(t)]
                for r in range(K): # brauch ich dann wahrscheindlich gar nicht mehr. 
                    for i0 in range(N):
                        for i1 in range(N):
                            for i2 in range(N):
                                elem += 0.5*2 * ( A[0, a, i0] *A[0, i0, i1] *A[0, i1, i2] *A[0, i2, c]*K4.get((a, i2), (i0, c), partA="c", partB="c", pairA=(0,2), pairB=(1,3))(t) #0123 
                                #+A[0, a, i0] *A[0, i0, i1] *A[0, i1, i2] *A[0, i2, c]*Cf(0,2, 1,3,  a, i1,  i1, c,partA="re",partB="im")(t) #0213
                                )

                # 2) erster Spur-Term: -1/2 * delta_{b d} * sum_{r,k} A_ak A_kc G_ck(t)
                if b == d:
                    for r in range(K):
                        for k in range(N):
                            elem -= 0.5*2 * (1

                            )

                # 3) zweiter Spur-Term: -1/2 * delta_{a c} * sum_{r,k} A_dk A_kb G_dk(t)
                if a == c:
                    for r in range(K):
                        for k in range(N):
                            elem -= 0.5*2 * (1

                            )

                if elem != 0.0:
                    rows.append(int(I))
                    cols.append(int(J))
                    data.append(elem)

        R_sparse = sp.coo_matrix(
            (np.array(data, dtype=complex),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(N**2, N**2),
            dtype=complex
        ).tocsr()

        R_qobj = Qobj(R_sparse, dims=dims)
        return L0 + R_qobj

    # QuTiP-kompatible Signatur (t, args)
    from functools import lru_cache
    t_round_L = 3 # 3 oder 4 ist ein guter Start (Tradeoff Speed/Genauigkeit)

    @lru_cache(maxsize=20000)
    def build_L_t_cached(t_key):
        return build_L_t(float(t_key))

    def L_of_t(t, args=None):
        return build_L_t_cached(round(float(t), t_round_L))

    return L_of_t


import qutip as qutip
P11=qutip.basis(2,1)*qutip.basis(2,1).dag()
P10=qutip.basis(2,1)*qutip.basis(2,0).dag()
P01=qutip.basis(2,0)*qutip.basis(2,1).dag()
P00=qutip.basis(2,0)*qutip.basis(2,0).dag()
psi0=qutip.basis(2,1)
rho0=P11
w0=1
T=1
lam=0.01
Td=1
gamma0=0.2
omega_c=10
spectrum_form="ohmic"
H=w0*P11
A=P10+P01
a_ops=[P10+P01]
e_ops=[P11]
tlist=np.linspace(0,10,16)
resutls = TCLsolve_TCL(
    H, psi0, tlist, a_ops,
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    e_ops, c_ops=None,
    options=None,
    use_TCL4=True,  # <-- wichtig
    use_secular=False, dw_min=0.0,
    sign_convention="plus",
)



def nbar(w, T):
        if T <= 0:
            return 0.0
        x = w / T
        if x < 1e-8:
            return 1.0 / x
        return 1.0 / (np.exp(x) - 1.0)




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


res_BR = qutip.brmesolve(H, rho0, tlist,a_ops=[[A,S_not_markov_hot]],e_ops=[P11])

rho11_red = np.array(res_BR.expect[0], dtype=float)



print(resutls)

pop = resutls.expect[0]
import os
import numpy as np


# Zielordner
outdir = "results"
os.makedirs(outdir, exist_ok=True)

# Dateiname (f-String!)
fname = f"TCL2_pop_TD{Td}_w{w0}_gamma{gamma0}.npy"

# korrekt speichern
np.save(
    os.path.join(outdir, fname),
    {"times": resutls.times, "pop": pop}
)

print("gespeichert unter:", os.path.join(outdir, fname))


print("gespeichert unter:", os.path.join(outdir, fname))

plt.plot(resutls.times, pop.real, marker="o")
plt.plot(tlist, rho11_red, label="BR rho11(t)", color="red")
plt.xlabel("t")
plt.ylabel(r"$\langle P_{11} \rangle$")
plt.title("TCL2 population")
plt.grid(True)
plt.show()
