import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# =========================
# Constants and parameters
# =========================
a = 1.0                      # lattice constant (set to 1)
t = -2.7                     # nearest-neighbor hopping (eV)
t_perp = 0.34                # interlayer hopping (AB bilayer)
vF = 3 * abs(t) * a / 2      # Fermi velocity (TB-derived)

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

# =========================
# Graphene lattice vectors
# =========================
delta = np.array([
    [0, -1],
    [np.sqrt(3)/2, 1/2],
    [-np.sqrt(3)/2, 1/2]
]) * a

# Reciprocal K point
K = np.array([4*np.pi/(3*np.sqrt(3)*a), 0])

# =========================
# k-path: Γ → K → M → Γ
# =========================
Gamma = np.array([0, 0])
M = np.array([np.pi/np.sqrt(3), np.pi/3])

def k_path(points, n=100):
    ks = []
    for i in range(len(points)-1):
        for s in np.linspace(0, 1, n):
            ks.append(points[i] + s*(points[i+1]-points[i]))
    return np.array(ks)

kpoints = k_path([Gamma, K, M, Gamma], n=80)

# =========================
# 1. Monolayer graphene TB
# =========================
def H_mono(k):
    f = sum(np.exp(1j * k @ d) for d in delta)
    return np.array([[0, t*f],
                     [t*np.conj(f), 0]])

bands_mono = np.array([eigh(H_mono(k))[0] for k in kpoints])

# =========================
# 2. AB-stacked bilayer TB
# =========================
def H_bilayer(k):
    f = sum(np.exp(1j * k @ d) for d in delta)
    H = np.zeros((4, 4), dtype=complex)

    # Layer 1
    H[0, 1] = t*f
    H[1, 0] = t*np.conj(f)

    # Layer 2
    H[2, 3] = t*f
    H[3, 2] = t*np.conj(f)

    # Interlayer AB coupling (B1-A2)
    H[1, 2] = t_perp
    H[2, 1] = t_perp

    return H

bands_bi = np.array([eigh(H_bilayer(k))[0] for k in kpoints])

# =========================
# 3. Twisted bilayer (TB-derived continuum)
# =========================
theta = 1.05 * np.pi / 180

# Moiré q-vectors
q1 = np.array([0, -2*K[0]*np.sin(theta/2)])
q2 = np.array([ np.sqrt(3)*K[0]*np.sin(theta/2), K[0]*np.sin(theta/2)])
q3 = np.array([-np.sqrt(3)*K[0]*np.sin(theta/2), K[0]*np.sin(theta/2)])

wAA = 0.079
wAB = 0.097

def Tmat(phi):
    return np.array([
        [wAA, wAB*np.exp(-1j*phi)],
        [wAB*np.exp(1j*phi), wAA]
    ])

T1 = Tmat(0)
T2 = Tmat(2*np.pi/3)
T3 = Tmat(-2*np.pi/3)

def H_dirac(k, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    q = R @ k
    return vF * (q[0]*sx + q[1]*sy)

def H_tbg(k):
    H = np.zeros((8, 8), dtype=complex)

    # Dirac blocks
    H[0:2, 0:2] = H_dirac(k, -theta/2)
    H[2:4, 2:4] = H_dirac(k + q1, -theta/2)
    H[4:6, 4:6] = H_dirac(k, theta/2)
    H[6:8, 6:8] = H_dirac(k + q1, theta/2)

    # Interlayer coupling
    H[0:2, 4:6] = T1
    H[4:6, 0:2] = T1.conj().T

    return H

# Small k-path around K
k_small = np.linspace(-0.05, 0.05, 120)
bands_tbg = np.array([eigh(H_tbg(np.array([kx, 0])))[0] for kx in k_small])

# =========================
# Plotting
# =========================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Monolayer
axs[0].plot(bands_mono, color='black')
axs[0].set_title("Monolayer Graphene")
axs[0].set_ylabel("Energy (eV)")

# Bilayer
axs[1].plot(bands_bi, color='black')
axs[1].set_title("AB Bilayer Graphene")

# Twisted bilayer
axs[2].plot(k_small, bands_tbg, color='black')
axs[2].set_title("Twisted Bilayer Graphene (1.05°)")
axs[2].set_xlabel("k")

plt.tight_layout()
plt.show()