import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

a = 2.46
gamma0 = 2.8

"""Define grid in reciprocal space"""
K = (4*np.pi)/(3*a)
N = 1000

kx = np.linspace(-K,K,N)
ky = np.linspace(-K,K,N)

KX, KY = np.meshgrid(kx, ky)

"""Define Nearest Neighbour sites"""
"""Choosing 0 as origin"""

sites = np.array([
    [0, a/(np.sqrt(3))], # Up
    [a/2, -a/(2*np.sqrt(3))], #Down Right
    [-a/2, -a/(2*np.sqrt(3))]]) #Down Left

def f(kx, ky, sites):
    sum = np.zeros_like(KX, dtype=complex)
    for site in sites:
        rx, ry = site
        sum += np.exp(1j * (kx * rx + ky * ry))
    return sum

"""Calculate positive and negative surfaces"""
E = gamma0 * np.abs(f(KX, KY, sites))
E_plus = E
E_neg = -E

"""Plot of positive and negative surfaces"""
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(KX, KY, E_plus, cmap='inferno', alpha=0.8,linewidth=0,antialiased=True)
ax.plot_surface(KX, KY, E_neg, cmap='viridis', alpha=0.8,linewidth=0, antialiased=True)

Kmag = 4*np.pi/(3*a)            # K = 4π/3a
k2 = 2*np.pi/(3*a)
k3 = 2*np.pi/(np.sqrt(3)*a)

hex_vertices = np.array([
    [ Kmag,  0.0, 0.0],
    [ k2,    k3,  0.0],
    [-k2,    k3,  0.0],
    [-Kmag,  0.0, 0.0],
    [-k2,   -k3,  0.0],
    [ k2,   -k3,  0.0]
])

hexagon = Poly3DCollection([hex_vertices],alpha=1,edgecolor='black',linewidth=2,antialiased=True)

ax.add_collection3d(hexagon)


ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-10, 10])

ax.set_xlabel('kx (1/Å)', fontsize=12)
ax.set_ylabel('ky (1/Å)', fontsize=12)
ax.set_zlabel('Energy (eV)', fontsize=12)

ax.view_init(elev=8, azim=10)
ax.set_box_aspect([1,1,1])
ax.margins(0.1)
plt.savefig('Monolayer_Graphene_Band_Structure.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

a = 2.46e-10  # Lattice constant in meters
hbar = 1.0545718e-34  # Planck's constant (J·s)
eV = 1.60217662e-19    # Electron volt (J)
vF = 10**6  # Fermi velocity (m/s)

qmax = 5e8
qx = np.linspace(-qmax, qmax, N)
qy = np.linspace(-qmax, qmax, N)
QX, QY = np.meshgrid(qx, qy)

QX_nm = QX * 1e-9
QY_nm = QY * 1e-9

E_small = hbar * vF * np.sqrt(QX**2 + QY**2) / eV
E_small_q_plus = E_small
E_small_q_neg = -E_small_q_plus


"""Plot of small q-space around Gamma point"""
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(QX_nm, QY_nm, E_small_q_plus, cmap='inferno', alpha=0.9, linewidth=0, antialiased=True)
ax.plot_surface(QX_nm, QY_nm, E_small_q_neg, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
ax.set_xlabel(r'$q_x$ (nm$^{-1}$)')
ax.set_ylabel(r'$q_y$ (nm$^{-1}$)')
ax.set_zlabel('Energy (eV)')

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

ax.view_init(elev=8, azim=10)
ax.set_box_aspect([1,1,1])
ax.margins(0.1)
plt.savefig('Monolayer_Graphene_Gamma_Point.png', bbox_inches='tight', pad_inches=0.1)
plt.show()


# Check slope at small q
mid = N//2
q = QX[mid, :]
E = E_small[mid,:]

m = int(0.1 * N)
q_fit = q[mid-m:mid+m]
E_fit = E[mid-m:mid+m]

x = np.abs(q_fit)
y = E_fit

slope, intercept = np.polyfit(x, y, 1)

v_F_calculated = slope * eV / hbar

print("Fitted slope (eV·m):", slope, "eV.m")
print("Calculated Fermi velocity (m/s):", v_F_calculated, "m/s")
print("Target Fermi velocity (m/s):", vF, "m/s")
print("Percent error=", 100*(v_F_calculated - vF)/vF, "%")