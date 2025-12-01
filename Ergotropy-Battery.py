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

N=39 # Größe Hilbertraum von vibronic system


#Bolzman verteilung funktion(frequenz, inv temp)
def nb(w, T):
    return np.float64(1 / (np.exp((w) /T) - 1))

print(nb(w0,T_h))

#######################################
#rho0
def rho0_build(N,N_Test_Level):
    rho0 = qutip.tensor(
    basis(2, 0) * basis(2, 0).dag(),
    basis(N, 0) * basis(N, 0).dag(),
    basis(N_Test_Level, 0) * basis(N_Test_Level, 0).dag(),
    )  # |e,0><e,0|
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



################################################################################
# Ergotropy function od time 



def ergotropy_ss(H,rho) -> float:
    # Eigenwerte von rho absteigend (r↓), Eigenwerte von H aufsteigend (ε↑)
    # Eigenwerte und Eigenvektoren Hamilton
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

def ergotropy_ss_diag_only(H,rho) -> float:
    # Eigenwerte von rho absteigend (r↓), Eigenwerte von H aufsteigend (ε↑)
    # Eigenwerte und Eigenvektoren Hamilton
    rho=rho.ptrace(2)
    diag_elements = rho.diag()
    # Diagonal-Matrix mit Nullen außenrum erzeugen (als NumPy)
    diag_matrix = np.diag(diag_elements)
    # Wieder zurück in ein Qobj konvertieren (damit es in QuTiP weiterverwendbar ist)
    rho_diag = qutip.Qobj(diag_matrix)
    #print(rho,rho_diag)

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

def Energy_ss(H,rho) -> float:
    rho_red=rho.ptrace(2)
    Energy=((rho_red*H).tr())
    return Energy


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
#Lev9
rho0_9=rho0_build(N,9)
H_9=Hamilton(N,9,w0,wcav,λ)
H_9_free=Hamilton_free_red(N,9,w0,wcav,λ)
c_ops_list_9=collaps_operators(N,9,T_h,T_c)


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
        (H_9, c_ops_list_9),
        #(H_10,c_ops_list_10),
        
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



ergo_2_ss=ergotropy_ss(H_2_free,RHOS_ss[0])
#ergo_2_ss_diag=ergotropy_ss_diag_only(H_2_free,RHOS_ss[0])
energy_2_ss=Energy_ss(H_2_free,RHOS_ss[0])


ergo_3_ss=ergotropy_ss(H_3_free,RHOS_ss[1])
#ergo_3_ss_diag=ergotropy_ss_diag_only(H_3_free,RHOS_ss[1])
energy_3_ss=Energy_ss(H_3_free,RHOS_ss[1])


ergo_4_ss=ergotropy_ss(H_4_free,RHOS_ss[2])
#ergo_4_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_4_ss=Energy_ss(H_4_free,RHOS_ss[2])

ergo_5_ss=ergotropy_ss(H_5_free,RHOS_ss[3])
#ergo_5_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_5_ss=Energy_ss(H_5_free,RHOS_ss[3])

ergo_6_ss=ergotropy_ss(H_6_free,RHOS_ss[4])
#ergo_5_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_6_ss=Energy_ss(H_6_free,RHOS_ss[4])

ergo_7_ss=ergotropy_ss(H_7_free,RHOS_ss[5])
#ergo_5_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_7_ss=Energy_ss(H_7_free,RHOS_ss[5])

ergo_8_ss=ergotropy_ss(H_8_free,RHOS_ss[6])
#ergo_5_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_8_ss=Energy_ss(H_8_free,RHOS_ss[6])

ergo_9_ss=ergotropy_ss(H_9_free,RHOS_ss[7])
#ergo_5_ss_diag=ergotropy_ss_diag_only(H_4_free,RHOS_ss[2])
energy_9_ss=Energy_ss(H_9_free,RHOS_ss[7])
#######################################################################################################################
#Plot

#bei ergotropy wird nicht jeder zeitschritt genommen. deswegen liste anpassen









n=np.array([2,3,4,5,6,7,8,9])

total_energy=[100,100,100,100,100,100,100,100]
filled_ergotropy=[ergo_2_ss/wcav*100, ergo_3_ss/(2*wcav)*100, ergo_4_ss/(3*wcav)*100,(ergo_5_ss/(4*wcav))*100,(ergo_6_ss/(5*wcav))*100,(ergo_7_ss/(6*wcav))*100,(ergo_8_ss/(7*wcav))*100,(ergo_9_ss/(8*wcav))*100]
filled_energy=[energy_2_ss/wcav*100, energy_3_ss/(2*wcav)*100, energy_4_ss/(3*wcav)*100,energy_5_ss/(4*wcav)*100,energy_6_ss/(5*wcav)*100,energy_7_ss/(6*wcav)*100,energy_8_ss/(7*wcav)*100,energy_9_ss/(8*wcav)*100]


figSäule, ax =plt.subplots(figsize=(11,6))

ax.bar(n,total_energy, edgecolor='black',facecolor='none')
ax.bar(2,filled_ergotropy[0],edgecolor='black',facecolor='darkviolet')
ax.bar(3,filled_ergotropy[1],edgecolor='black',facecolor='blue')
ax.bar(4,filled_ergotropy[2],edgecolor='black',facecolor='cyan')
ax.bar(5,filled_ergotropy[3],edgecolor='black',facecolor='springgreen')
ax.bar(6,filled_ergotropy[4],edgecolor='black',facecolor='gold')
ax.bar(7,filled_ergotropy[5],edgecolor='black',facecolor='orange')
ax.bar(8,filled_ergotropy[6],edgecolor='black',facecolor='tomato')
ax.bar(9,filled_ergotropy[7],edgecolor='black',facecolor='red')

transpa=0.26
ax.bar(2,filled_energy[0],edgecolor='black',facecolor='darkviolet',alpha=transpa)
ax.bar(3,filled_energy[1],edgecolor='black',facecolor='blue',alpha=transpa)
ax.bar(4,filled_energy[2],edgecolor='black',facecolor='cyan',alpha=transpa)
ax.bar(5,filled_energy[3],edgecolor='black',facecolor='springgreen',alpha=transpa)
ax.bar(6,filled_energy[4],edgecolor='black',facecolor='gold',alpha=transpa)
ax.bar(7,filled_energy[5],edgecolor='black',facecolor='orange',alpha=transpa)
ax.bar(8,filled_energy[6],edgecolor='black',facecolor='tomato',alpha=transpa)
ax.bar(9,filled_energy[7],edgecolor='black',facecolor='red',alpha=transpa)

ax.set_ylabel(r"$\frac{\mathcal{W}_{B}(t)} {  n \omega_B}$",fontsize=23,rotation=0,labelpad=35  )
ax.text(-0.090, 0.80, r"$\frac{\mathrm{Tr}[\mathcal{H}_{B free} \rho(t)]}{ n \omega_B }$", transform=ax.transAxes,rotation=0,fontsize=23, va='bottom', ha='center')
ax.set_xlabel(
    r"Spinsystem-size $n$",
    fontsize=20,
    rotation=0,
    labelpad=20  # Abstand von der Achse
)
for i in range(8):
    #ax.bar(n[i], filled_ergotropy[i], edgecolor='black', facecolor=colors[i])
    
    # Prozent berechnen
    percent = (filled_ergotropy[i] / total_energy[i]) * 100
    
    # Text über den gefüllten Balken
    ax.text(
        n[i], 
        filled_ergotropy[i] + total_energy[i] * 0.03,  # leichter Abstand nach oben
        f'{percent:.1f}%', 
        ha='center', va='bottom', fontsize=16, fontweight='bold'
    )

for i in range(8):
    #ax.bar(n[i], filled_ergotropy[i], edgecolor='black', facecolor=colors[i])
    
    # Prozent berechnen
    percent = (filled_energy[i] / total_energy[i]) * 100
    
    # Text über den gefüllten Balken
    ax.text(
        n[i], 
        filled_energy[i] + total_energy[i] * 0.03,  # leichter Abstand nach oben
        f'{percent:.1f}%', 
        ha='center', va='bottom', fontsize=17, fontweight='bold'
    )
import matplotlib.ticker as mticker

# X-Achse: nur ganze Zahlen anzeigen, keine Kommas
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
ax.tick_params(labelsize=20)
ax.set_yticks([])         # entfernt die Tick-Marker
ax.set_yticklabels([])    # entfernt die Tick-Beschriftungen
ax.tick_params(left=False)  # entfernt die Tick-Linien


#legend = ax.legend(loc='upper left', frameon=True)
#legend.get_frame().set_facecolor('white')
plt.savefig("Battery.png", dpi=400, bbox_inches="tight")
plt.show()
