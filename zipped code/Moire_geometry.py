import numpy as np
import matplotlib.pyplot as plt

# Rotation matrix
def rot(phi):
    a, b = np.cos(phi), np.sin(phi)
    return np.array([[a, -b], [b, a]])

# Plotting moire geometry for theta = 2.3 degrees
def plot_moire(theta_deg=2.3, a_nm=0.246):
    theta_rad = np.radians(theta_deg)

    # Primitive lattice vectors
    a1 = a_nm * np.array([0.5, np.sqrt(3)/2])
    a2 = a_nm * np.array([-0.5, np.sqrt(3)/2])

    # carbon-carbon length --> defining atom A at (0,0) and then atom B at (0,a_cc)
    a_cc = a_nm / np.sqrt(3)
    tauA = np.array([0.0, 0.0])
    tauB = np.array([0.0, a_cc])

    # Apply rotation to each layer
    R1 = rot(-theta_rad/2)
    R2 = rot(theta_rad/2)

    # Calculating Moiré Superlattice
    L_M = a_nm / (2 * np.sin(theta_rad/2))

    aM1 = L_M * np.array([0.5,  np.sqrt(3)/2])
    aM2 = L_M * np.array([-0.5, np.sqrt(3)/2])

    # Real-space coordinates
    W = 1.25 * L_M

    N = int(np.ceil(W/a_nm)) + 8
    ms = np.arange(-N, N+1)
    ns = np.arange(-N, N+1)

    # Building lattice arrays
    ptsA, ptsB = [], []
    for m in ms:
        for n in ns:
            Rmn = m * a1 + n * a2
            ptsA.append(Rmn + tauA)
            ptsB.append(Rmn + tauB)

    ptsA = np.array(ptsA)
    ptsB = np.array(ptsB)

    # Apply rotation matrices to unrotated coordinates
    A1 = (R1 @ ptsA.T).T
    A2 = (R2 @ ptsB.T).T
    B1 = (R1 @ ptsB.T).T
    B2 = (R2 @ ptsA.T).T

    # Define cropped window for array sizes
    def in_window(P, W):
        return (np.abs(P[:, 0]) <= W) & (np.abs(P[:, 1]) <= W)

    A1, B1 = A1[in_window(A1, W)], B1[in_window(B1, W)]
    A2, B2 = A2[in_window(A2, W)], B2[in_window(B2, W)]

    # Scatter plot of all A, B atoms in layers 1 and 2
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.scatter(A1[:, 0], A1[:, 1], s=6, alpha=0.75, label="Layer 1 (A)")
    ax.scatter(B1[:, 0], B1[:, 1], s=6, alpha=0.75, label="Layer 1 (B)")
    ax.scatter(A2[:, 0], A2[:, 1], s=6, alpha=0.35, label="Layer 2 (A)")
    ax.scatter(B2[:, 0], B2[:, 1], s=6, alpha=0.35, label="Layer 2 (B)")

    # Define corners of moiré unit cell
    Origin = -0.5*(aM1 + aM2)
    P1 = Origin + aM1
    P2 = Origin + aM2
    P12 = Origin + aM1 + aM2
    cell = np.vstack([Origin, P1, P12, P2, Origin])

    cell_xy = cell[:-1]
    xmin, ymin = cell_xy.min(axis=0)
    xmax, ymax = cell_xy.max(axis=0)

    margin = 0.25 * L_M
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    # Plot moiré unit cell
    ax.plot(cell[:, 0], cell[:, 1], 'k-', lw=2, label="Moiré Unit Cell")

    mid_a1 = Origin + 0.5 * aM1
    mid_a2 = Origin + 0.5 * aM2
    u1 = aM1 / np.linalg.norm(aM1)
    u2 = aM2 / np.linalg.norm(aM2)
    perp1 = np.array([-u1[1], u1[0]])
    perp2 = np.array([-u2[1], u2[0]])
    offset = 0.08 * L_M
    ax.text(*(mid_a1 + offset*perp1), r"$\mathbf{a}_1^{M}$", fontsize=16, color='blue')
    ax.text(*(mid_a2 + offset*perp2), r"$\mathbf{a}_2^{M}$", fontsize=16, color='blue')

    ax.set_xlabel("x (nm)", fontsize=14)
    ax.set_ylabel("y (nm)", fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig("moire_geometry_tbg_primitive_vectors.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print(f"Twist angle θ = {theta_deg:.4f}°")
    print(f"Moiré lattice constant L_M = {L_M:.3f} nm")

#plot_moire(theta_deg=2.3, a_nm=0.246)
