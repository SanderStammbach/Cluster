from qutip import basis, sigmax, sigmaz, sigmay, Options
import qutip
from qutip import ptrace
import numpy as np
from qutip import steadystate as steadystate
from qutip import tensor as tensor
import Self_made_solver 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from concurrent.futures import ProcessPoolExecutor


#######Parameter##############
  # With hbar = 1 this is the upper energy
v=1
w0=100

λ = 2.5  # dim less coupling constant depending on the shift of the energy minima shifts in space
wcav=w0-2*λ**2*v
g = 0.3 # couplig cavity to sysem
gamma_h = 0.01
gamma_c = 0.1
T_h = w0/(np.log(2))   # temp Einheit von  hb*omega durch kb

T_c = 1.2

N=30 # Größe Hilbertraum von vibronic system

tlist = np.linspace(0, 3000, 2000 )

#Bolzman verteilung funktion(frequenz, inv temp)
def nb(w, T):
    return np.float64(1 / (np.exp((w) /T) - 1))

print(nb(w0,T_h))

#######################################
#rho0


def rho0_build(N, N_Test_Level):
    mix_strength=0.5 
    coherence_strength=0.3
    rhoA = basis(2, 0) * basis(2, 0).dag()
    rhoB = basis(N, 0) * basis(N, 0).dag()
    k0 = basis(N_Test_Level, 0)
    # Mischung mit anderen Zuständen: hier einfache Superposition mit |1>
    if N_Test_Level > 1:
        k1 = basis(N_Test_Level, 1)
        # Baue 2D-Unterraum-Zustände mit off-diagonalen
        rhoC = (1 - mix_strength) * (k0 * k0.dag()) \
             + mix_strength * (k1 * k1.dag()) \
             + coherence_strength * np.sqrt(mix_strength * (1 - mix_strength)) * (k0 * k1.dag() + k1 * k0.dag())
    else:
        # Wenn nur 1 Level existiert - reiner ..
        rhoC = k0 * k0.dag()
    # --- Hermitisieren & normieren ---
    rhoC = 0.5 * (rhoC + rhoC.dag())
    rhoC = rhoC / rhoC.tr()
    # --- Gesamtes Tensorprodukt ---
    rho0 = qutip.tensor(rhoA, rhoB, rhoC)

    return rho0

########################################
### Operators & Hamilton  as function ###############

def operators(N,N_Test_Level):
    j=(N_Test_Level-1)/2
    b = qutip.tensor(
    qutip.qeye(2), qutip.destroy(N), qutip.qeye(N_Test_Level)
    )  # ladder operator of the vibronic modes
    c = qutip.tensor(
    qutip.qeye(2), qutip.qeye(N), qutip.spin_Jp(j)
    )  # ladder operator of the photonic modes
    cd = qutip.tensor(qutip.qeye(2), qutip.qeye(N), qutip.spin_Jm(j))
    P12 = qutip.tensor(
    qutip.spin_Jp(1/2), qutip.qeye(N), qutip.qeye(N_Test_Level)
    )
    P21 = qutip.tensor(
    qutip.spin_Jm(1/2), qutip.qeye(N), qutip.qeye(N_Test_Level)
    )
    P11 = qutip.tensor(
    basis(2, 0) * basis(2, 0).dag(), qutip.qeye(N), qutip.qeye(N_Test_Level)
    )
    P22 = qutip.tensor(
    basis(2, 1) * basis(2, 1).dag(), qutip.qeye(N), qutip.qeye(N_Test_Level)
    )
    S_z = qutip.tensor(
    j*qutip.qeye(N_Test_Level)-qutip.spin_Jz(j), qutip.qeye(N), qutip.qeye(N_Test_Level)
    )
    C_z  = qutip.tensor(qutip.qeye(2), qutip.qeye(N), j*qutip.qeye(N_Test_Level)-qutip.spin_Jz(j))
    C22  = qutip.tensor(qutip.qeye(2), qutip.qeye(N),basis(2, 1) * basis(2, 1).dag())

    return b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22



def Hamilton(N,N_Test_Level,w0,wcav,λ):
    b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22=operators(N,N_Test_Level)
    
    return v * b.dag() * b + w0*P21*P12 + λ * v * (b + b.dag()) * P21*P12 + wcav *C_z + g * (P21 * c + cd*P12 ) #Normaler Hamilton



def Hamilton_free_red(N,N_Test_Level,w0,wcav,λ):
   j=(N_Test_Level-1)/2
   C_z_red=j*qutip.qeye(N_Test_Level)-qutip.spin_Jz(j)
    
   return  wcav *C_z_red #Normaler Hamilton

#Function Collaps 
def collaps_operators(N,N_Test_Level,T_h,T_c):
    b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22=operators(N,N_Test_Level)
    c_op_list = []

    c_op_list.append(np.sqrt((nb(w0, T_h) + 1) * gamma_h) * P12)
    c_op_list.append(np.sqrt((nb(w0, T_h)) * gamma_h) * P21)
    c_op_list.append(np.sqrt((nb(v, T_c) + 1) * gamma_c) * (b+λ*P21*P12) )
    c_op_list.append(np.sqrt((nb(v, T_c)) * gamma_c) * (b.dag()+λ*P12*P21) )
    return c_op_list
##########################################################################




################################################################################
# Ergotropy function od time 

def ergotropy_t(H, rho_t, timelist): 
    # Eigenwerte von rho absteigend (r↓), Eigenwerte von H aufsteigend (ε↑)
    # Eigenwerte und Eigenvektoren des Hamiltonians

    e_vals, e_vecs = H.eigenstates()        
    zipH = zip(e_vals, e_vecs)
    sorted_zipH = sorted(zipH, key=lambda z: z[0], reverse=False)
    sorted_e_vals, sorted_e_vecs = zip(*sorted_zipH)
    #rho_t_part=[rho.ptrace(2) for rho in rho_t]
    ergotropy_list = []
    print(len(rho_t),len(timelist))
    
    for idx, t in enumerate(timelist):
        rho_full = rho_t[idx]
        rho_state= rho_full.ptrace(2)

        # Eigenwerte und Eigenvektoren von ρ
        r_vals, r_vecs = rho_state.eigenstates()   
        zipRho = zip(r_vals, r_vecs)
        sorted_zipRho = sorted(zipRho, key=lambda z: z[0], reverse=True)
        sorted_r_vals, sorted_r_vecs = zip(*sorted_zipRho)

        rhoPI = 0
        for n in range(len(sorted_e_vecs)):
            rhoPI += sorted_r_vals[n] * sorted_e_vecs[n] * sorted_e_vecs[n].dag()
        
        # Tr(ρH) und Tr(ρπH)
        UrhoPI = (rhoPI * H).tr()
        Urho   = (rho_state * H).tr()
        ergotropy_list.append(Urho - UrhoPI)

    return ergotropy_list

# Ergotropy function of steady state rho_ss


def ergotropy_ss(H,rho) -> float:
    # Eigenwerte von rho absteigend (r↓), Eigenwerte von H aufsteigend (ε↑)
    # Eigenwerte und Eigenvektoren Hamilton
    #rho=rho.ptrace(2)

    N_test=len(H.dims)
    rho=rho.ptrace(2)
    e_vals, e_vecs = H.eigenstates()        # Energies
    zipH=zip(e_vals,e_vecs)
    sorted_zipH=sorted(zipH,key=lambda zipH: zipH[0], reverse=False )
    sorted_e_vals, sorted_e_vecs=zip(*sorted_zipH)

    # Eigenwerte und Eigenvektoren Rho
    r_vals, r_vecs = rho.eigenstates()   
    zipRho=zip(r_vals,r_vecs)
    sorted_zipRho=sorted(zipRho,key=lambda zipRho: zipRho[0], reverse=True)
    sorted_r_vals, sorted_r_vecs=zip(*sorted_zipRho)
    rhoPI=0
    for n in range(len(sorted_e_vecs)):
        rhoPI+=sorted_r_vals[n]*sorted_e_vecs[n]*sorted_e_vecs[n].dag()
    # Tr(rho H)
    UrhoPI=((rhoPI * H).tr())
    Urho = ((rho * H).tr())

    return Urho - UrhoPI



#########################################
#Define Hamilons of different Demon levels 
#Lev2
rho0_2=rho0_build(N,2)
H_2=Hamilton(N,2,w0,wcav,λ)
H_2_free=Hamilton_free_red(N,2,w0,wcav,λ)
c_ops_list_2=collaps_operators(N,2,T_h,T_c)
#Lev3
rho0_3=rho0_build(N,3)
H_3=Hamilton(N,3,w0,wcav,λ)
H_3_free=Hamilton_free_red(N,3,w0,wcav,λ)
c_ops_list_3=collaps_operators(N,3,T_h,T_c)
#Lev4
rho0_4=rho0_build(N,4)
H_4=Hamilton(N,4,w0,wcav,λ)
H_4_free=Hamilton_free_red(N,4,w0,wcav,λ)
c_ops_list_4=collaps_operators(N,4,T_h,T_c)
#Lev5
rho0_5=rho0_build(N,5)
H_5=Hamilton(N,5,w0,wcav,λ)
H_5_free=Hamilton_free_red(N,5,w0,wcav,λ)
c_ops_list_5=collaps_operators(N,5,T_h,T_c)
#Lev6
rho0_6=rho0_build(N,6)
H_6=Hamilton(N,6,w0,wcav,λ)
H_6_free=Hamilton_free_red(N,6,w0,wcav,λ)
c_ops_list_6=collaps_operators(N,6,T_h,T_c)
#Lev7
rho0_7=rho0_build(N,7)
H_7=Hamilton(N,7,w0,wcav,λ)
H_7_free=Hamilton_free_red(N,7,w0,wcav,λ)
c_ops_list_7=collaps_operators(N,7,T_h,T_c)
#Lev8
rho0_8=rho0_build(N,8)
H_8=Hamilton(N,8,w0,wcav,λ)
H_8_free=Hamilton_free_red(N,8,w0,wcav,λ)
c_ops_list_8=collaps_operators(N,8,T_h,T_c)
"""#Lev9
rho0_9=rho0_build(N,9)
H_9=Hamilton(N,9,w0,wcav,λ)
H_9_free=Hamilton_free_red(N,9,w0,wcav,λ)
c_ops_list_9=collaps_operators(N,9,T_h,T_c)
#Lev10
rho0_10=rho0_build(N,10)
H_10=Hamilton(N,10,w0,wcav,λ)
H_10_free=Hamilton_free(N,10,w0,wcav,λ)
c_ops_list_10=collaps_operators(N,10,T_h,T_c)"""

##############################################################################################################
##############################################################################################################
                                            # Check Hilbert space size #

def check_oscillator_overflow(rho, N, N_Test_Level, p_tol=1e-4, sigma_factor=3):
    """
    Check harmonischer Oszillator (Trunkierung N)
    zu stark angeregt wurde.

    Args:
      rho      : Qobj Dichtematrix (oder Ket) für das Oszillator-Subsystem.
      N        : Trunkierungs-Dimension (Anzahl Fock-Level).
      p_tol    : Schwelle für Populationsanteil im höchsten Level.
      sigma_factor: Faktor für Standardabweichung-Check.

    Returns:
      overflow : bool, True wenn potenzieller Overflow erkannt.
      info     : dict mit Diagnosedaten.
    """
    # 1) Wahrscheinlichkeiten der Fock-Zustände
    #    rho.diag() liefert die Diagonale in Fock-Basis, falls rho Dichteoperator.
    pops = np.real(rho.diag())

    # 2) Anteil im obersten und vorletzten Level
    p_top = pops[-1]
    p_second = pops[-2] if N > 1 else 0.0
    tail_prob = p_top + p_second

    # 3) Mittlere Besetzungszahl und Varianz
    n_op = qutip.tensor(
        qutip.qeye(2), qutip.num(N), qutip.qeye(N_Test_Level)
    )  # number operator (N x N)
    n_mean = qutip.expect(n_op, rho)
    n_var = qutip.expect(n_op**2, rho) - n_mean**2
    n_sigma = np.sqrt(abs(n_var))

    # 4) Overflow-Kriterien
    overflow = False
    reasons = []

    if tail_prob > p_tol:
        overflow = True
        reasons.append(
            f"Großer Rand-Populationsanteil: P[N-2]+P[N-1]={tail_prob:.2e} > {p_tol:.1e}"
        )
    if n_mean + sigma_factor * n_sigma > N - 1:
        overflow = True
        reasons.append(
            f"<n>+{sigma_factor}σ = {n_mean:.2f}+{sigma_factor}·{n_sigma:.2f} ≈ {n_mean+n_sigma*sigma_factor:.2f} ≥ {N-1}"
        )

    info = {
        "pops": pops,
        "p_top": p_top,
        "p_second": p_second,
        "tail_prob": tail_prob,
        "n_mean": n_mean,
        "n_sigma": n_sigma,
        "overflow_reasons": reasons,
    }
    return overflow, info
rho_ss = steadystate(H_2, c_ops_list_2)
overflow, info = check_oscillator_overflow(rho_ss, N,2)
if overflow:
    print("⚠️ Overflow-Warnung!  N erhöhen.")
    for reason in info["overflow_reasons"]:
        print("   •", reason)
else:
    print("✅ Trunkierung scheint ausreichend (kein Overflow).")



#####################################################################################################################
#####################################################################################################################
# Solve rho_t 
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
#import qutip as qt

# Schnelles Ausgabeverzeichnis (lokaler Scratch ist ideal)
OUTDIR = Path(os.environ.get("SCRATCH", ".")) / f"ergotropy_{os.getpid()}"
OUTDIR.mkdir(parents=True, exist_ok=True)

def run_solve_to_disk(task_id, H, rho0, tlist, c_ops):
    print(f"run {task_id}")
    states = Self_made_solver.solve_liouvillian_expm(H, rho0, tlist, c_ops,Auslassen=3)
    fn = OUTDIR / f"states_{task_id}.qo"     # Endung beliebig; qsave macht Pickle
    qutip.qsave(states, str(fn))
    return str(fn)                           # nur kleiner String zurück

def _unpack(args):
    return run_solve_to_disk(*args)

if __name__ == "__main__":
    # Threads zähmen (wichtig gegen Oversubscription)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tasks = [
        (2, H_2, rho0_2, tlist, c_ops_list_2),
        (3, H_3, rho0_3, tlist, c_ops_list_3),
        (4, H_4, rho0_4, tlist, c_ops_list_4),
        (5, H_5, rho0_5, tlist, c_ops_list_5),
        (6, H_6, rho0_6, tlist, c_ops_list_6),
        (7, H_7, rho0_7, tlist, c_ops_list_7),
        (8, H_8, rho0_8, tlist, c_ops_list_8),
        #(9, H_9, rho0_9, tlist, c_ops_list_9),
    ]

    # workers = 3 wie bei dir; optional: chunksize=1
    with ProcessPoolExecutor(max_workers=3) as ex:
        state_files = list(ex.map(_unpack, tasks, chunksize=1))

    # Laden Alle################################################################
# Laden Alle ################################################################
results = [qutip.qload(path) for path in state_files]


#################################################################################################
#################################################################################################
# Steady state 
def run_steadystate(H,c_ops):
    return steadystate(H,c_ops)

def _unpack_ss(args):
    return run_steadystate(*args)   # entpackt (H, rho0, tlist, c_ops, e_ops)


if __name__ == "__main__":

    # Drei verschiedene Aufgaben
    tasks = [
        (H_2, c_ops_list_2),
        (H_3, c_ops_list_3),
        (H_4, c_ops_list_4),
        (H_5, c_ops_list_5),
        (H_6, c_ops_list_6),
        (H_7, c_ops_list_7),
        (H_8, c_ops_list_8),
        #(H_9, c_ops_list_9),
        
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        RHOS_ss = list(executor.map(_unpack_ss, tasks))

################################################################################################
"""overflow, info = check_oscillator_overflow(results[0].states[0.5], N,N_Test_Level)
if overflow:
    print("⚠️ Overflow-Warnung!  N erhöhen.")
    for reason in info["overflow_reasons"]:
        print("   •", reason)
else:
    print("✅ Trunkierung scheint ausreichend (kein Overflow).")   
"""

####################################################################################################
                        #ergotropy berechnen



#immer time evolution und steadystate
tlist_plot=[t for i, t in enumerate(tlist) if i % 3 ==0]


ergo_2_t = ergotropy_t(H_2_free, results[0], tlist_plot)
ergo_2_ss=ergotropy_ss(H_2_free,RHOS_ss[0])

ergo_3_t=ergotropy_t(H_3_free,results[1],tlist_plot)
ergo_3_ss=ergotropy_ss(H_3_free,RHOS_ss[1])

ergo_4_t=ergotropy_t(H_4_free,results[2],tlist_plot)
ergo_4_ss=ergotropy_ss(H_4_free,RHOS_ss[2])

ergo_5_t=ergotropy_t(H_5_free,results[3],tlist_plot)
ergo_5_ss=ergotropy_ss(H_5_free,RHOS_ss[3])

ergo_6_t=ergotropy_t(H_6_free,results[4],tlist_plot)
ergo_6_ss=ergotropy_ss(H_6_free,RHOS_ss[4])

ergo_7_t=ergotropy_t(H_7_free,results[5],tlist_plot)
ergo_7_ss=ergotropy_ss(H_7_free,RHOS_ss[5])

ergo_8_t=ergotropy_t(H_8_free,results[6],tlist_plot)
ergo_8_ss=ergotropy_ss(H_8_free,RHOS_ss[6])

"""ergo_9_t=ergotropy_t(H_9_free,results[7],tlist_plot)
ergo_9_ss=ergotropy_ss(H_9_free,RHOS_ss[7])"""

                
#######################################################################################################################
#Plot

#bei ergotropy wird nicht jeder zeitschritt genommen. deswegen liste anpassen

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#Missmatch der länge beheben


ax.plot(tlist_plot, ergo_2_t, label=r"$n=2$", color="darkviolet")
ax.hlines(ergo_2_ss, xmin=tlist[0], xmax=tlist[-1], color="darkviolet", linestyle="--")
ax.axvline(x=1/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(1/gamma_h, 1.02, r"$\frac{1}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)




ax.plot(tlist_plot, ergo_3_t, label=r"$n=3$", color="blue")
ax.hlines(ergo_3_ss, xmin=tlist[0], xmax=tlist[-1], color="blue", linestyle="--")
ax.axvline(x=2/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(2/gamma_h, 1.02, r"$\frac{2}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)

ax.plot(tlist_plot, ergo_4_t, label=r"$n=4$", color="cyan")
ax.hlines(ergo_4_ss, xmin=tlist[0], xmax=tlist[-1], color="cyan", linestyle="--")
ax.axvline(x=3/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(3/gamma_h, 1.02, r"$\frac{3}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)

ax.plot(tlist_plot, ergo_5_t, label=r"$n=5$", color="springgreen")
ax.hlines(ergo_5_ss, xmin=tlist[0], xmax=tlist[-1], color="springgreen", linestyle="--")
ax.axvline(x=4/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(4/gamma_h, 1.02, r"$\frac{4}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)

ax.plot(tlist_plot, ergo_6_t, label=r"$n=6$", color="gold")
ax.hlines(ergo_6_ss, xmin=tlist[0], xmax=tlist[-1], color="gold", linestyle="--")
ax.axvline(x=5/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(5/gamma_h, 1.02, r"$\frac{5}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)

ax.plot(tlist_plot, ergo_7_t, label=r"$n=7$", color="orange")
ax.hlines(ergo_7_ss, xmin=tlist[0], xmax=tlist[-1], color="orange", linestyle="--")
ax.axvline(x=6/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(6/gamma_h, 1.02, r"$\frac{6}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)

ax.plot(tlist_plot, ergo_8_t, label=r"$n=8$", color="tomato")
ax.hlines(ergo_8_ss, xmin=tlist[0], xmax=tlist[-1], color="tomato", linestyle="--")
ax.axvline(x=7/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(7/gamma_h, 1.02, r"$\frac{7}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)


"""ax.plot(tlist_plot, ergo_9_t, label=r"$n=9$", color="red")
ax.hlines(ergo_9_ss, xmin=tlist[0], xmax=tlist[-1], color="red", linestyle="--")
ax.axvline(x=8/gamma_h, color='k', linestyle='--',alpha=0.6)
ax.text(8/gamma_h, 1.02, r"$\frac{8}{\gamma_h}$", transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='black', fontsize=14)"""


ax.set_xscale('log')
ax.set_xlim(left=1e1)

ax.set_xlabel(r"$ t \nu$",fontsize=14)
ax.set_ylabel(
    r"$\frac{\mathcal{W}_{\mathcal{H}_{free}}(t)} { \nu}$",
    fontsize=14,
    rotation=0,
    labelpad=20  # Abstand von der Achse
)

#ax.set_title("Ergotropy als Funktion der Zeit")


ax.grid(True, which="both", linestyle=":", linewidth=0.8)
legend = ax.legend(loc='upper left', frameon=True)
legend.get_frame().set_facecolor('white')
plt.savefig("Plot_Ergotropy_Hfree-N_2-8.png",dpi=300, bbox_inches="tight")
plt.show()


"""
def pop_check(rho_t):
    rand1=[]
    rand2=[]
    rand3=[]
    for i,t in enumerate(rho_t):
        rho=rho_t[i]
        diag_werte= np.real(rho.diag())
        rand1.append(np.abs( diag_werte[-1]))
        rand2.append( np.abs(diag_werte[-2]))
        rand3.append( np.abs(diag_werte[-3]))
    return rand1 , rand2 ,rand3
 
rand1,rand2,rand3=pop_check(results[0])
fig2, ax =plt.subplots(1,1,figsize=(6,5))
ax.plot(tlist_plot, rand1, color="darkgreen",label="R1")
ax.plot(tlist_plot, rand2, color="forestgreen",label="R2")
ax.plot(tlist_plot, rand3, color="lime",label="R3")
ax.grid(True, which="both", linestyle=":", linewidth=0.8)
legend = ax.legend(loc='upper left', frameon=True)
legend.get_frame().set_facecolor('white')
ax.set_xlabel(r"$ t \nu$",fontsize=14)
ax.set_ylabel(
    "Randpopulation",
    fontsize=14,
    rotation=90,
    labelpad=20  # Abstand von der Achse
)
plt.savefig("Error.svg", dpi=300, bbox_inches="tight")
plt.show()

"""
