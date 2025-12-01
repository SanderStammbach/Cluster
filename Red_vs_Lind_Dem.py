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
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

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
# --- Anfangszustand: angeregt ---


# --- Zeitachse ---
tlist = np.linspace(0, 1000, 1000)

N=30 # Größe Hilbertraum von vibronic system


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
# --- Anfangszustand: angeregt ---


# --- Zeitachse ---

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
N_Test_Level=2


b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22=operators(N,N_Test_Level)

# --- TLS-Operatoren ---
  # sigma_+
sigma_x = P12 + P21
B=b+b.dag()+2*λ*P12*P21
# --- Parameter ---


def nbar(w, T):
    # stabil für skalare w
    if np.isclose(w, 0.0):
        return 1e6
    return 1.0 / (np.exp(w / T) - 1.0)

# --- Hamiltonoperator (E0=0, E1=ω0) ---




# === Lindblad mit thermischen Raten ===

c_ops=collaps_operators(N,N_Test_Level,T_h,T_c)
H=Hamilton(N,N_Test_Level,w0,wcav,λ)
rho0=rho0_build(N,N_Test_Level)
res_L = qt.mesolve(H, rho0, tlist, c_ops, e_ops=[P22,C22])

# === Bloch-Redfield mit "deltaförmigem" Markov-Spektrum ===
# S(ω0) = γ (n̄+1), S(-ω0) = γ n̄, sonst 0. -> liefert exakt dieselben Raten wie oben.
tolh = 50
def S_markov_hot(w):
    if np.isclose(w,  w0, atol=tolh):
        return gamma_h * (nbar(w, T_h) + 1.0)
    if np.isclose(w, -w0, atol=tolh):
        return gamma_h * nbar(-w, T_h)
    return 0.0
tolc=0.8
def S_markov_cold(w):
    if np.isclose(w,  v, atol=tolc):
        return gamma_c * (nbar(w, T_c) + 1.0)
    if np.isclose(w, -v, atol=tolc):
        return gamma_c * nbar(-w, T_c)
    return 0.0
###############################################################################
w_min = 1e-6   # schützt vor nbar-Divergenz bei w->0

def S_not_markov_hot(w):
    omega_c = 500
    wabs = max(abs(w), w_min)

    # exponentieller Cutoff, bei w0 ~ O(1) in etwa 1 und ansonsten <=1
    cutoff = np.exp(-(wabs - w0) / (omega_c))
    cutoff = min(cutoff, 1.0)  # verhindert Verstärkung für w < w0

    if w > 0:
        return gamma_h * (nbar(wabs, T_h) + 1.0) * cutoff
    elif w < 0:
        return gamma_h * nbar(wabs, T_h) * cutoff
    else:
        return 0.0


def S_not_markov_cold(w):
    omega_c = 5
    wabs = max(abs(w), w_min)

    cutoff =  np.exp(-(wabs - v) / (omega_c))
    cutoff = min(cutoff, 1.0)

    if w > 0:
        return gamma_c * (nbar(wabs, T_c) + 1.0) * cutoff
    elif w < 0:
        return gamma_c * nbar(wabs, T_c) * cutoff
    else:
        return 0.0

###################################################################################
opts = Options(
    nsteps=1000,   # Standard ist 1000 – deutlich erhöhen
    atol=1e-8,
    rtol=1e-6
)
res_BR = qt.brmesolve(
    H, rho0, tlist,
    #a_ops=[(sigma_x, S_not_markov),(sigma_x, S_markov)], fuer zweites bad
    a_ops=[sigma_x,B],
    e_ops=[P22,C22],
    spectra_cb=[S_not_markov_hot, S_not_markov_cold ],
    options=opts
      # Säkularapproximation -> Lindblad-Form
)

# --- Plot ---
plt.figure(figsize=(9,6))
plt.plot(tlist, res_L.expect[0] ,color='green', label=r" Lindblad: $ P_e^S $ ")
plt.plot(tlist, res_BR.expect[0], '--', color='green',label=r" Bloch-Redfield: $ P_e^S $ ")
plt.plot(tlist, res_L.expect[1],color='red', label=r" Lindblad: $ P_e^D $ ")
plt.plot(tlist, res_BR.expect[1], '--',color='red', label=r" Bloch-Redfield: $ P_e^D $ ")
plt.xlabel(r"$t \nu $",fontsize=20)
plt.ylabel(r"$ P_e $",rotation=0, labelpad=25, fontsize=20)
plt.tick_params(labelsize=17)
#plt.title('Identischer Zerfall: Lindblad = Bloch-Redfield (gematchtes Spektrum)')
plt.legend(fontsize=17)
plt.grid(True)
plt.savefig("Red_final.png", dpi=400, bbox_inches="tight")
plt.show()

