import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = 2.46
gamma0 = 3
hbar = 1.0545718e-34  # Planck's constant (J·s)
eV = 1.60217662e-19    # Electron volt (J)
vF = 10**6  # Fermi velocity (m/s)

# Reciprocal grid
K = (4*np.pi)/(3*a)
N = 1000

kx = np.linspace(-K, K, N)
ky = np.linspace(-K, K, N)
KX, KY = np.meshgrid(kx, ky)

kmax = 0.1 * K
kx_small = np.linspace(-kmax, kmax, N)
ky_small = np.linspace(-kmax, kmax, N)
KX_small, KY_small = np.meshgrid(kx_small, ky_small)

def H_bilayer(kx, ky):
    H = [[]]