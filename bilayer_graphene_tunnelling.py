import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = 2.46
gamma0 = 3
theta = 1.1 * np.pi / 180  # Twist angle in radians
h_bar = 6.582119569e-16  # Planck's constant in eV*s
vF = (np.sqrt(3) * a * gamma0) / (2 * h_bar)  # Fermi velocity in m/s


# Reciprocal grid
K = np.array([(4*np.pi)/(3*a),0])
Nk = 25

kx = np.linspace(-K[0], K[0], Nk)
ky = np.linspace(-K[1], K[1], Nk)
KX, KY = np.meshgrid(kx, ky)

# No twist angle

def H_mono(kx,ky):
    
    H = -gamma0 * np.array([[0, np.sqrt(3)*a/2*(kx - 1j*ky)],[np.sqrt(3)*a/2*(kx + 1j*ky), 0]])
    return H

wAA = 0.0975
wAB = 0.0797

phis = [0, 2*np.pi/3, -2*np.pi/3]

T_mats = [
    np.array([[wAA, wAB * np.exp(-1j * phi)], [wAB * np.exp(1j * phi), wAA]]) for phi in phis
]

def rotation_matrix(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

# Rotate K points

K1 = rotation_matrix(theta/2) @ K
K2 = rotation_matrix(-theta/2) @ K

q1 = K2 - K1
q2 = rotation_matrix(2*np.pi/3) @ q1
q3 = rotation_matrix(-2*np.pi/3) @ q1

q = [q1, q2, q3]
alpha = wAB / (h_bar * vF * np.linalg.norm(q1))
print(alpha)
 
N_G = 6
G_list= []
b1 = q2-q1
b2 = q3-q1

for m in range(-N_G, N_G+1):
    for n in range(-N_G, N_G+1):
        G_list.append(m*b1 + n*b2)
G_list = np.array(G_list)
NG = len(G_list)

def H_BM(kx,ky,theta):
    
    dimensions = 4 * NG
    H = np.zeros((dimensions, dimensions), dtype=complex)

    for i, G in enumerate(G_list):
        kG = np.array([kx, ky]) - G

        k1 = rotation_matrix(theta/2) @ kG  # Layer 1
        k2 = rotation_matrix(-theta/2) @ kG # Layer 2

        H1 = H_mono(k1[0], k1[1])
        H2 = H_mono(k2[0], k2[1])

        H[4*i:4*i+2, 4*i:4*i+2] = H1
        H[4*i+2:4*i+4, 4*i+2:4*i+4] = H2

    for i, G in enumerate(G_list):
        for j, Gp in enumerate(G_list):
            for qs, T in zip(q, T_mats):
                if np.linalg.norm(G - Gp - qs) < 1e-6:  # Check if G and Gp are connected by q
                    H[4*i:4*i+2, 4*j+2:4*j+4] = T
                    H[4*j+2:4*j+4, 4*i:4*i+2] = T.conj().T  # Hermitian conjugate

    return H

bands = np.zeros((Nk, Nk, 4*NG))

for i in range(Nk):
    for j in range(Nk):
        H = H_BM(KX[i, j], KY[i, j], theta)
        eigenvalues = LA.eigh(H)[0]  # Only return eigenvalues, not eigenvectors
        bands[i, j, :] = eigenvalues

# Define  high symmetry points

Gamma = np.array([0, 0])
K_point = (2*b1 + b2)/3 # Corner of hexagon
M = (b1 + b2)/2 # Midpoint of edge of hexagon

def interpolate(k1, k2, Nk):
    return np.array([k1 + (k2-k1)*t for t in np.linspace(0, 1, Nk)])

Nk_path = 40

k_path = np.vstack([
    interpolate(Gamma, K_point, Nk_path),
    interpolate(K_point, M, Nk_path),
    interpolate(M, Gamma, Nk_path)
])

x_ticks = [0, Nk_path, 2*Nk_path, 3*Nk_path]
x_labels = ['Γ', 'K', 'M', 'Γ']

bands = []

for kx, ky in k_path:
    H = H_BM(kx, ky, theta)
    eigenvalues = LA.eigh(H)[0]
    bands.append(eigenvalues)

bands = np.array(bands)

plt.figure(figsize=(8, 6))
for n in range(bands.shape[1]):
    plt.plot(bands[:, n], color='blue', linewidth=0.8)
for xc in x_ticks:
    plt.axvline(x=xc, color='red', linestyle='--', linewidth=0.5)
plt.xticks(x_ticks, x_labels)
plt.xlabel('Path in k-space')
plt.ylabel('Energy (eV)')
plt.title('Band structure of bilayer graphene with twist angle')
plt.show()


Nk = 20
k_range = 0.05

kx = np.linspace(-k_range, k_range, Nk)
ky = np.linspace(-k_range, k_range, Nk)
KX, KY = np.meshgrid(kx, ky)

nbands = 8
bands2d = np.zeros((Nk, Nk, nbands))

for i in range(Nk):
    for j in range(Nk):
        H = H_BM(KX[i, j], KY[i, j], theta)
        eigenvalues = LA.eigh(H)[0]
        bands2d[i, j, :] = eigenvalues[:nbands]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(KX, KY, bands2d[:, :, 0], cmap='viridis', edgecolor='none')

ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_zlabel('Energy (eV)')
ax.set_title('Lowest Moiré Band near Γ')

plt.tight_layout()
plt.show()