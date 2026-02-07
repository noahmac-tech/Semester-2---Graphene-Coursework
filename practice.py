import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh

a = 2.46
gamma0 = 3


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

"""Plot on same 3D axes"""
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