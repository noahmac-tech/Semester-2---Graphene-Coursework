import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh

a = 2.46
gamma0 = 3
hbar = 1.0545718e-34  # Planck's constant (J·s)
eV = 1.60217662e-19    # Electron volt (J)
vF = 10**6  # Fermi velocity (m/s)


"""Define grid in reciprocal space"""
K = (4*np.pi)/(3*a)
N = 1000

kx = np.linspace(-K,K,N)
ky = np.linspace(-K,K,N)

KX, KY = np.meshgrid(kx, ky)

"""Small k space sites"""
kmax = 0.1 * K
kx_small = np.linspace(-kmax, kmax, N)
ky_small = np.linspace(-kmax, kmax, N)
KX_small, KY_small = np.meshgrid(kx_small, ky_small)


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

E_small = hbar * vF * np.sqrt(KX_small**2 + KY_small**2) / eV
E_small_plus = E_small
E_small_neg = -E_small

"""Plot of positive and negative surfaces"""
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(KX, KY, E_plus, cmap='inferno', alpha=0.8)
ax.plot_surface(KX, KY, E_neg, cmap='viridis', alpha=0.8)
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-10, 10])

ax.set_xlabel('kx (1/Å)', fontsize=12)
ax.set_ylabel('ky (1/Å)', fontsize=12)
ax.set_zlabel('Energy (eV)', fontsize=12)

ax.view_init(elev=8, azim=10)
plt.show()

"""Plot of small k-space around K point"""
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(KX_small, KY_small, E_small_plus, cmap='inferno', alpha=0.9)
ax.plot_surface(KX_small, KY_small, E_small_neg, cmap='viridis', alpha=0.9)

ax.set_xlabel(r'$k_x$ (m$^{-1}$)')
ax.set_ylabel(r'$k_y$ (m$^{-1}$)')
ax.set_zlabel('Energy (eV)')
ax.set_title('Monolayer Graphene Band Structure Near K (Dirac Cone)')
ax.view_init(elev=8, azim=10)
plt.tight_layout()
plt.show()