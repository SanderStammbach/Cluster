
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

def make_G_from_Gamma(Gamma_plus, Gamma_minus, W):
    def Gp(r, i, j, t):
        return Gamma_plus(t, float(W[i, j]))

    def Gm(r, i, j, t):
        return Gamma_minus(t, float(W[i, j]))

    return Gp, Gm

def TCLsolve_TCL2(
    H, psi0, tlist, a_ops,
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    e_ops, c_ops=None,
    options=None,
    use_secular=False, dw_min=0.0,
    sign_convention="plus",
):
    if e_ops is None:
        e_ops = []
    if c_ops is None:
        c_ops = []
    if options is None:
        options = Options()

    # 1) Basis-Liouvillian (Hamilton + evtl. c_ops)
    L0 = liouvillian(H, c_ops)  # Qobj (N^2 x N^2)

    # 2) Eigenbasis: ekets, Frequenzen W, Operatoren A[r,i,j]
    evals, ekets = H.eigenstates()
    N = len(evals)
    W = np.subtract.outer(evals, evals)   # W[i,j] = E_i - E_j

    K = len(a_ops)
    A = np.zeros((K, N, N), dtype=complex)
    for r, Ar in enumerate(a_ops):
        Ar_eb = Ar.transform(ekets)       # in Eigenbasis
        A[r,:,:] = Ar_eb.full()

    # 3) Iabs: alle (I,a,b)
    Iabs = np.array([(a + N*b, a, b) for a in range(N) for b in range(N)], dtype=np.int32)

    # 4) Gamma aus Spektraldichte
    C, Gamma_plus, Gamma_minus = gamma_maker_with_J(
        T=T, w0=w0, lam=lam, Td=Td, gamma0=gamma0, omega_c=omega_c,
        spectrum_form=spectrum_form,
        tau_grid=tlist,   # <-- wichtig: hier rein!
    )

    Gp, Gm = make_G_from_Gamma(Gamma_plus, Gamma_minus, W)


    # 6) L(t) bauen
    L_of_t = build_L_TCL2_time_dependent(
        L0=L0, A=A, W=W, Iabs=Iabs, K=K, N=N,
        G=Gp ,Gc=Gm, use_secular=use_secular, dw_min=dw_min
    )

    # 7) Zeitabhängig integrieren
    results = TCL_solve_time_dep(L_of_t, ekets, psi0, tlist, e_ops=e_ops, options=options)

    if e_ops:
        return TCLResult("tcl2", tlist, expect=results, states=None)
    else:
        return TCLResult("tcl2", tlist, expect=None, states=results)






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



from scipy.integrate import quad
import numpy as np

def gamma_maker_with_J(
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    tau_grid,
    wmin=1e-3, wmax=100.0,
    epsabs=1e-8, epsrel=1e-6, limit=10000,
):
    beta = 1.0 / float(T)

    def coth_scalar(x):
        x = float(x)
        if abs(x) < 1e-8:
            return 1.0 / x
        return 1.0 / np.tanh(x)

    # ---- C(tau) per quad (cos/sin weighted) ----
    def C_quad(tau):
        tau = float(tau)

        def f_re(w):
            return J(w, w0, lam, Td, gamma0, omega_c, spectrum_form) * coth_scalar(0.5 * beta * w)

        def f_im(w):
            return J(w, w0, lam, Td, gamma0, omega_c, spectrum_form)

        Cr, _ = quad(
            f_re, wmin, wmax, weight="cos", wvar=tau,
            epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        Ci, _ = quad(
            f_im, wmin, wmax, weight="sin", wvar=tau,
            epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return Cr - 1j * Ci

    # ---- tabelliere C einmal auf tau_grid ----
    tau_grid = np.asarray(tau_grid, float)
    C_vals = np.array([C_quad(tau) for tau in tau_grid], dtype=complex)
    Cc_vals = np.conjugate(C_vals)

    # ---- Gamma(t,Omega) Tabellen (Cache pro Omega) ----
    Gamma_cache = {}  # key: ("+", Omega) oder ("-", Omega)

    def _Gamma_table(Omega, which):
        Omega = float(Omega)
        key = (which, Omega)
        if key in Gamma_cache:
            return Gamma_cache[key]

        if which == "+":
            integrand = C_vals * np.exp(1j * Omega * tau_grid)
        elif which == "-":
            integrand = Cc_vals * np.exp(-1j * Omega * tau_grid)
        else:
            raise ValueError("which must be '+' or '-'")

        dt = np.diff(tau_grid)
        Gtab = np.zeros_like(tau_grid, dtype=complex)
        Gtab[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dt)

        Gamma_cache[key] = Gtab
        return Gtab

    def Gamma_plus(t, Omega):
        tab = _Gamma_table(Omega, "+")
        return np.interp(float(t), tau_grid, tab)

    def Gamma_minus(t, Omega):
        tab = _Gamma_table(Omega, "-")
        return np.interp(float(t), tau_grid, tab)

    return C_quad, Gamma_plus, Gamma_minus





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
Td=3.2
gamma0=0.02
omega_c=10
spectrum_form="ohmic"
H=w0*P11
A=P10+P01
a_ops=[P10+P01]
e_ops=[P11]
tlist=np.linspace(0,20,10000)
resutls=TCLsolve_TCL2(
    H, psi0, tlist, a_ops,
    T, w0, lam, Td, gamma0, omega_c, spectrum_form,
    e_ops, c_ops=None,
    options=None,
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
"""
# korrekt speichern
np.save(
    os.path.join(outdir, fname),
    {"times": resutls.times, "pop": pop}
)"""

print("gespeichert unter:", os.path.join(outdir, fname))


print("gespeichert unter:", os.path.join(outdir, fname))

plt.plot(resutls.times, pop.real, marker="o")
plt.plot(tlist, rho11_red, label="BR rho11(t)", color="red")
plt.xlabel("t")
plt.ylabel(r"$\langle P_{11} \rangle$")
plt.title("TCL2 population")
plt.grid(True)
plt.show()