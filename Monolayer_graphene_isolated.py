import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_full_bz_surface(a_ang, gamma0, N):
    """3D tight-binding dispersion over a square k-grid spanning ±K."""
    K = (4*np.pi)/(3*a_ang)

    kx = np.linspace(-K, K, N)
    ky = np.linspace(-K, K, N)
    KX, KY = np.meshgrid(kx, ky)

    r_nn = a_ang/np.sqrt(3)
    sites = np.array([
        [0.0, r_nn],                 # up
        [a_ang/2, -r_nn/2],           # down-right
        [-a_ang/2, -r_nn/2],          # down-left
    ])

    def f(kx, ky):
        s = np.zeros_like(KX, dtype=complex)
        for rx, ry in sites:
            s += np.exp(1j * (kx*rx + ky*ry))
        return s

    E = gamma0 * np.abs(f(KX, KY))
    E_plus, E_neg = E, -E

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(KX, KY, E_plus, cmap='inferno', alpha=0.8, linewidth=0, antialiased=True)
    ax.plot_surface(KX, KY, E_neg,  cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

    Kmag = 4*np.pi/(3*a_ang)
    k2 = 2*np.pi/(3*a_ang)
    k3 = 2*np.pi/(np.sqrt(3)*a_ang)
    hex_vertices = np.array([
        [ Kmag,  0.0, 0.0],
        [ k2,    k3,  0.0],
        [-k2,    k3,  0.0],
        [-Kmag,  0.0, 0.0],
        [-k2,   -k3,  0.0],
        [ k2,   -k3,  0.0],
    ])
    hexagon = Poly3DCollection([hex_vertices], alpha=1, edgecolor='black', linewidth=2, antialiased=True)
    ax.add_collection3d(hexagon)

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-10, 10])
    ax.set_xlabel('kx (1/Å)', fontsize=12)
    ax.set_ylabel('ky (1/Å)', fontsize=12)
    ax.set_zlabel('Energy (eV)', fontsize=12)
    ax.view_init(elev=8, azim=10)
    ax.set_box_aspect([1, 1, 1])
    ax.margins(0.1)

    plt.tight_layout()
    plt.savefig('Monolayer_Graphene_Band_Structure.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_dirac_cone(qmax, N, vF):
    """3D Dirac cone in small-q approximation around a Dirac point."""
    hbar = 1.0545718e-34
    eV = 1.60217662e-19

    qx = np.linspace(-qmax, qmax, N)
    qy = np.linspace(-qmax, qmax, N)
    QX, QY = np.meshgrid(qx, qy)

    QX_nm = QX * 1e-9
    QY_nm = QY * 1e-9

    E = hbar * vF * np.sqrt(QX**2 + QY**2) / eV
    E_plus, E_neg = E, -E

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(QX_nm, QY_nm, E_plus, cmap='inferno', alpha=0.9, linewidth=0, antialiased=True)
    ax.plot_surface(QX_nm, QY_nm, E_neg,  cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)

    ax.set_xlabel(r'$q_x$ (nm$^{-1}$)')
    ax.set_ylabel(r'$q_y$ (nm$^{-1}$)')
    ax.set_zlabel('Energy (eV)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=8, azim=10)
    ax.set_box_aspect([1, 1, 1])
    ax.margins(0.1)

    plt.tight_layout()
    plt.savefig('Monolayer_Graphene_Gamma_Point.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def check_slope(qmax, N, vF):
    """Calculating slope at dirac point to verify it matches vF."""
    hbar = 1.0545718e-34
    eV = 1.60217662e-19

    qx = np.linspace(-qmax, qmax, N)
    qy = np.linspace(-qmax, qmax, N)
    QX, QY = np.meshgrid(qx, qy)

    E = hbar * vF * np.sqrt(QX**2 + QY**2) / eV

    mid = N//2
    q = QX[mid, :]
    Eline = E[mid, :]

    m = int(0.1 * N)
    q_fit = q[mid-m:mid+m]
    E_fit = Eline[mid-m:mid+m]

    coeffs, cov = np.polyfit(np.abs(q_fit), E_fit, 1, cov=True)
    
    slope = coeffs[0]
    intercept = coeffs[1]

    slope_error = np.sqrt(cov[0,0])
    
    
    v_F_calculated = slope * eV / hbar  # Convert to m/s
    
    print(f"Calculated vF ≈ {v_F_calculated:.2e} m/s")
    print("Error=", slope_error)


#plot_full_bz_surface(a_ang=2.46, gamma0=2.8, N=1000)
#plot_dirac_cone(qmax=5e8, N=1000, vF=1e6)
check_slope(qmax=5e8, N=800, vF=1e6)