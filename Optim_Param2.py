from qutip import basis, sigmax, sigmaz, sigmay, Options
import qutip
from qutip import ptrace
import numpy as np
from qutip import steadystate as steadystate
from qutip import tensor as tensor
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patheffects
from concurrent.futures import ProcessPoolExecutor


#######Parameter##############
  # With hbar = 1 this is the upper energy
v=1 # Ich rechne mit v 
w0=100

#λ = 2.5  # dim less coupling constant depending on the shift of the energy minima shifts in space
#wcav=w0-2*λ**2*v
g = 0.3 # couplig cavity to sysem
gamma_h = 0.01
gamma_c = 0.1
T_h = w0/(np.log(2))   # temp Einheit von  hb*omega durch kb
T_c = 1.2

N=35 # Größe Hilbertraum von vibronic system



#Bolzman verteilung funktion(frequenz, inv temp)
def nb(w, T):

    return (np.float128(1 / (np.exp((w) /T) - 1)))



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

#print(basis(2,1)*basis(2,0).dag()*basis(2,0)*basis(2,1).dag(),qutip.spin_Jm(1/2)*qutip.spin_Jp(1/2))

def Hamilton(N,N_Test_Level,w0,wcav,λ,g):
    b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22=operators(N,N_Test_Level)
    
    return v * b.dag() * b + w0*P21*P12 + λ * v * (b + b.dag()) * P21*P12 + wcav *C_z + g * (P21 * c + cd*P12 ) #Normaler Hamilton


#Function Collaps 
def collaps_operators(N,N_Test_Level,T_h,T_c,gamma_c,λ,wcav):
    b ,c, cd, P12,P21,P11,P22,S_z,C_z,C22=operators(N,N_Test_Level)
    c_op_list = []

    c_op_list.append(np.sqrt((nb(w0, T_h) + 1) * gamma_h) * P12)
    c_op_list.append(np.sqrt((nb(w0, T_h)) * gamma_h) * P21)
    c_op_list.append(np.sqrt((nb(v, T_c) + 1) * gamma_c) * (b+λ*P21*P12) )
    c_op_list.append(np.sqrt((nb(v, T_c)) * gamma_c) * (b.dag()+λ*P12*P21) )
    #c_op_list.append(np.sqrt((nb(wcav, 1) + 1) * gamma_c) * (c) )
    #c_op_list.append(np.sqrt((nb(wcav, 1)) * gamma_c) * (cd) )
    return c_op_list

def grid_maker():
    vec1 = np.linspace(0, 7, 260,dtype=np.float64)                     # y-Achse (Δ)
    vec2 = np.linspace(-12, 12 ,260,dtype=np.float64)  # x-Achse (λj)
    Z = np.empty((len(vec1), len(vec2)))                  # Z[i,j] = (vec1[i], vec2[j])

    for λj in enumerate(vec1):
        print("sowit",λj[0],λj[1])
        for Δi in enumerate(vec2):
            Δ=Δi[1]
            λ=λj[1]
            
            wcav=w0-Δ*(λ**2)*v
            H = Hamilton(N, 2, w0, wcav, λ, g)
            c_ops = collaps_operators(N, 2, T_h, T_c, gamma_c,λ,wcav)
            b, c, cd, P12, P21, P11, P22, S_z, C_z, C22 = operators(N, 2)
            #if g < 0:#10**(-5.5):
            #    rho_ss = qutip.steadystate(H,c_ops, method='svd',atol=1e-20,rtol=1e-20,maxiter=10000)
            #else :
            rho_ss=qutip.steadystate(H, c_ops)
            X = (C22 * rho_ss).tr()
            Z[λj[0], Δi[0]] = float(np.real(X))
    return Z, vec2, vec1


Z, vec2, vec1 = grid_maker()  # Z.shape = (len(vec1), len(vec2))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterMathtext
# Beispiel-Daten (ersetze mit deinen)
# vec1, vec2 = np.logspace(-6, 1, 200), np.logspace(-5, -2, 200)
# Z = np.random.rand(len(vec1), len(vec2))  # Dummy-Daten

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)

# Titel
ax.set_title(r"$\langle \rho_{ee} \rangle$", fontsize=20)

# 2D-Gitter
X, Y = np.meshgrid(vec2, vec1)

# Farbskala zentriert auf 0.5
norm = colors.TwoSlopeNorm(vmin=Z.min(), vcenter=0.5, vmax=Z.max())

# Plot
pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap='PRGn')#,norm=norm)


class ScaledLogFormatter(LogFormatterMathtext):
    def __init__(self, scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def __call__(self, x, pos=None):
        # Beschriftung so, als wäre die Achse mit Faktor "scale" multipliziert
        return super().__call__(x * self.scale, pos)

# 100× größere Tickbeschriftung
#ax.xaxis.set_major_formatter(ScaledLogFormatter(scale=100))

# Log-Skalen
#ax.set_xscale('log')
#ax.set_yscale('log')

# Achsenbeschriftungen
ax.set_xlabel(r"$ \Delta $", fontsize=30)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*1:g}"))
ax.set_ylabel(r"$\lambda$", fontsize=30, rotation=0, labelpad=40)
ax.tick_params(axis='both', labelsize=20)

# Contour bei 0.5
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s} "

cs = ax.contour(X, Y, Z, levels=[0.5], colors='black', linestyles='--')
ax.clabel(cs, cs.levels, fmt=fmt, fontsize=15)

# Achsenverhältnis quadratisch
ax.set_box_aspect(1)

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig.tight_layout()
fig.canvas.draw()  # Positionen aktualisieren

# Aktuelle Achsenposition (in Figure-Koordinaten)
pos = ax.get_position()

# Etwas Platz rechts vom Plot schaffen
pad = 0.03       # horizontaler Abstand (in Figure-Koordinaten)
width = 0.035      # Breite der Colorbar (in Figure-Koordinaten)
shrink = 0.96     # wie viel Prozent der Achsenhöhe die Bar haben soll (optisch perfekt)

# Colorbar exakt ausgerichtet an Achse
cax = fig.add_axes([
    pos.x1 + pad,                                  # Start rechts vom Plot
    pos.y0 + (1 - shrink) * pos.height / 2,        # zentriert, leicht kürzer
    width,                                         # Breite
    pos.height * shrink                            # Höhe
])

cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=20)
# cbar.set_label(r"$\langle \rho_{ee} \rangle$", fontsize=14)

plt.savefig("OccupationProb_nanopart.png",dpi=800, bbox_inches="tight", pad_inches=0.2)
plt.show()







