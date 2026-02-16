import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = 2.46
hbar = 1.0545718e-34  # Planck's constant (J·s)
eV = 1.60217662e-19    # Electron volt (J)
vF = 10**6  # Fermi velocity (m/s)

gamma1 = 0.39 * eV
gamma3 = 0.315 * eV

v3 = (np.sqrt(3) * a * gamma3) / (2 * hbar)

# Reciprocal grid
K = (4*np.pi)/(3*a)
N = 200

kx = np.linspace(-K, K, N)
ky = np.linspace(-K, K, N)
KX, KY = np.meshgrid(kx, ky)

bands = np.zeros((4, N, N), dtype=complex)
bands_trigonal = np.zeros((4, N, N), dtype=complex)

kmax = 5e8 # 1/m, small region around K point
kx_small = np.linspace(-kmax, kmax, N)
ky_small = np.linspace(-kmax, kmax, N)
KX_small, KY_small = np.meshgrid(kx_small, ky_small)

def H_bilayer(kx_small, ky_small):
    alpha = hbar * (kx_small + 1j * ky_small)
    alpha_conj = np.conj(alpha)
    H = [
         [0, vF * alpha, 0, 0],
         [vF*alpha_conj, 0, gamma1, 0],
         [0, gamma1, 0, vF* alpha_conj],
         [0, 0, vF*alpha, 0]
        ]
    return np.array(H)

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
    for j in range(N):
        eigs = LA.eigvalsh(H_bilayer(KX_small[i, j], KY_small[i, j]))
        bands[:,i,j] = eigs /eV

for i in range(N):
    for j in range (N):
        eigs_trig = LA.eigvalsh(H_bilayer_trigonal_warping(KX_small[i, j], KY_small[i, j])) 
        bands_trigonal[:,i,j] = eigs_trig / eV


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(KX_small, KY_small, bands[0], cmap='viridis', alpha=0.8) 
ax.plot_surface(KX_small, KY_small, bands[1], cmap='inferno', alpha=0.8) 
ax.plot_surface(KX_small, KY_small, bands[2], cmap='plasma', alpha=0.8)
ax.plot_surface(KX_small, KY_small, bands[3], cmap='cividis', alpha=0.8) 

ax.set_xlabel('kx (1/m)')
ax.set_ylabel('ky (1/m)')
ax.set_zlabel('Energy (eV)')

ax.set_title('Bilayer Graphene Band Structure Near K')
ax.view_init(elev=0, azim=45)
plt.show()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(KX_small, KY_small, bands_trigonal[0], cmap='viridis', alpha=0.8) 
ax.plot_surface(KX_small, KY_small, bands_trigonal[1], cmap='inferno', alpha=0.8) 
ax.plot_surface(KX_small, KY_small, bands_trigonal[2], cmap='plasma', alpha=0.8) 
ax.plot_surface(KX_small, KY_small, bands_trigonal[3], cmap='cividis', alpha=0.8) 
ax.set_xlabel('qx (1/m)') 
ax.set_ylabel('qy (1/m)') 
ax.set_zlabel('Energy (eV)') 
ax.set_title('Bilayer Graphene Band Structure with Trigonal Warping') 
ax.view_init(elev=0, azim=45) 
plt.show()






