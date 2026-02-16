import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = 2.46e-10
hbar = 1.0545718e-34  # Planck's constant (J·s)
eV = 1.60217662e-19    # Electron volt (J)
vF = 10**6  # Fermi velocity (m/s)

gamma1 = 0.39 * eV
gamma3 = 0.315 * eV

v3 = (np.sqrt(3) * a * gamma3) / (2 * hbar)
print(v3)

# Reciprocal grid
K = (4*np.pi)/(3*a)
N = 400

kx = np.linspace(-K, K, N)
ky = np.linspace(-K, K, N)
KX, KY = np.meshgrid(kx, ky)

bands = np.zeros((4, N, N), dtype=complex)
bands_trigonal = np.zeros((4, N, N), dtype=complex)

kmax = 1e5 # 1/m, small region around K point

kx_small = np.linspace(-kmax, kmax, N)
ky_small = np.linspace(-kmax, kmax, N)
KX_small, KY_small = np.meshgrid(kx_small, ky_small)

def H_bilayer_trigonal_warping(qx, qy):
    pi = hbar * (qx + 1j*qy)
    pi_dag = np.conj(pi)

    return np.array([
        [0,        vF*pi_dag,  0,        v3*pi],
        [vF*pi,    0,          gamma1,   0],
        [0,        gamma1,     0,        vF*pi_dag],
        [v3*pi_dag,0,          vF*pi,    0]
    ], dtype=complex)


for i in range(N):
    for j in range (N):
        eigs_trig = LA.eigvalsh(H_bilayer_trigonal_warping(KX_small[i, j], KY_small[i, j])) 
        bands_trigonal[:,i,j] = eigs_trig / eV

plt.figure()
plt.contour(KX_small, KY_small, bands_trigonal[1], levels=40)
plt.gca().set_aspect('equal')
plt.title("Trigonal Warping - Constant Energy Contours")
plt.xlabel(r'$q_x$ (m$^{-1}$)')
plt.ylabel(r'$q_y$ (m$^{-1}$)')
plt.colorbar(label='Energy (eV)')
plt.show()