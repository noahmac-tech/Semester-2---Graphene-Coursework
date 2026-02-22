import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.constants import hbar, eV
   

a = 2.46e-10

gamma1 = 0.39 * eV
gamma3 = 0.315 * eV

vF = 10**6  # Fermi velocity (m/s)


"""Define grid in reciprocal space"""
qmax = 5e8 # 1/m, small region around K point
N = 200

qx = np.linspace(-qmax, qmax, N)
qy = np.linspace(-qmax, qmax, N)
QX, QY = np.meshgrid(qx, qy)

QX_nm = QX * 1e-9
QY_nm = QY * 1e-9

bands = np.zeros((4, N, N), dtype=float)

def H_bilayer(qx, qy):
    H = [
        [0, vF * hbar * (qx + 1j * qy), 0, 0],
        [vF * hbar * (qx - 1j * qy), 0, gamma1, 0],
        [0, gamma1, 0, vF * hbar * (qx - 1j * qy)],
        [0, 0, vF * hbar * (qx + 1j * qy), 0]
    ]
    return np.array(H)

#Diagonalise Hamiltonian at each k-point
for i in range(N):
    for j in range(N):
        eigs_J = LA.eigvalsh(H_bilayer(QX[i, j], QY[i, j]))
        bands[:,i,j] = eigs_J / eV  # Convert to eV

mid = N // 2  # qy=0 index (since linspace symmetric)
plt.figure(figsize=(7,4))
for n in range(4):
    plt.plot(QX_nm[mid, :], bands[n, mid, :], label=f'Band {n+1}')
plt.xlabel(r'$q_x$ (nm$^{-1}$) at $q_y=0$')
plt.ylabel('Energy (eV)')
plt.ylim(-0.5, 0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Bilayer_Graphene_Bands_qy0.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

plt.figure(figsize=(6,5))
cs = plt.contourf(QX_nm, QY_nm, bands[2], levels=40)  # pick conduction low-energy band
plt.xlabel(r'$q_x$ (nm$^{-1}$)')
plt.ylabel(r'$q_y$ (nm$^{-1}$)')
plt.colorbar(cs, label='Energy (eV)')
plt.axis('equal')
plt.tight_layout()
plt.savefig('Bilayer_Graphene_Contour.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot the 4 bands
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(QX_nm, QY_nm, bands[0], cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
ax.plot_surface(QX_nm, QY_nm, bands[1], cmap='inferno', alpha=0.5, linewidth=0, antialiased=True)
ax.plot_surface(QX_nm, QY_nm, bands[2], cmap='plasma', alpha=0.5, linewidth=0, antialiased=True)
ax.plot_surface(QX_nm, QY_nm, bands[3], cmap='cividis', alpha=0.5, linewidth=0, antialiased=True)

ax.set_xlabel(r'$q_x$ (nm$^{-1}$)')
ax.set_ylabel(r'$q_y$ (nm$^{-1}$)')
ax.set_zlabel('Energy (eV)', fontsize=12)

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

ax.view_init(elev=8, azim=10)
ax.set_box_aspect([1,1,1])
ax.margins(0.1)
plt.savefig('Bilayer_Graphene_Band_Structure.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
