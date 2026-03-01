import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

a = 2.46
d = 3.35
gamma0 = 2.8
t0 = 0.35
r0 = 0.45

r_nn = a / np.sqrt(3)
nn_cut = 1.10 * r_nn
interlayer_cut = 5.0

a1 = a * np.array([ 1/2,  np.sqrt(3)/2])
a2 = a * np.array([-1/2,  np.sqrt(3)/2])

tauA = np.array([0.0, 0.0])
tauB = np.array([0.0, r_nn])

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def theta_from_mn(m, n):
    num = m**2 + 4*m*n + n**2
    den = 2*(m**2 + m*n + n**2)
    cos_t = np.clip(num/den, -1, 1)
    return np.arccos(cos_t)

def supercell_vectors(m, n):
    t1 = m*a1 + n*a2
    t2 = -n*a1 + (m+n)*a2
    return t1, t2

def fractional_coors(points, t1, t2):
    M = np.column_stack((t1, t2))
    invM = np.linalg.inv(M)
    return points @ invM.T

def wrap_to_cell(points, t1, t2):
    uv = fractional_coors(points, t1, t2)
    uv_wrapped = uv - np.floor(uv)
    return uv_wrapped @ np.column_stack((t1, t2)).T

def generate_layer1_in_cell(m, n):
    t1, t2 = supercell_vectors(m, n)

    span = (m + n) + 2
    ii = np.arange(-span, span+1)
    jj = np.arange(-span, span+1)

    pts = []
    subl = []

    for i in ii:
        for j in jj:
            R = i*a1 + j*a2
            pts.append(R + tauA); subl.append(0)
            pts.append(R + tauB); subl.append(1)

    pts = np.array(pts)
    subl = np.array(subl, dtype=int)

    uv = fractional_coors(pts, t1, t2)
    mask = (uv[:,0] >= 0) & (uv[:,0] < 1) & (uv[:,1] >= 0) & (uv[:,1] < 1)
    
    pts_cell = pts[mask]
    subl_cell = subl[mask]

    return pts_cell, subl_cell, t1, t2

def build_sparse_H_tbg(pos1, subl1, pos2, subl2, t1, t2):

    N1 = len(pos1)
    N2 = len(pos2)
    N = N1 + N2

    rows, cols, data = [], [], []

    shifts = []
    for u in [-1, 0, 1]:
        for v in [-1, 0, 1]:
            shifts.append(u*t1 + v*t2)
    shifts = np.array(shifts)

    pos1_img = (pos1[None,:,:] + shifts[:,None,:]).reshape(-1, 2)
    idx1_img = np.tile(np.arange(N1), len(shifts))

    
    tree1 = cKDTree(pos1_img)
    for i in range(N1):
        neigh = tree1.query_ball_point(pos1[i], nn_cut)
        for p in neigh:
            j = idx1_img[p]
            if j == i:
                continue
            if subl1[i] != subl1[j]:
                if i < j:
                    rows += [i, j]
                    cols += [j, i]
                    data += [-gamma0, -gamma0]
    
    
    pos2_img = (pos2[None,:,:] + shifts[:,None,:]).reshape(-1,2)
    idx2_img = np.tile(np.arange(N2), len(shifts))

    tree2 = cKDTree(pos2_img)
    for i in range(N2):
        neigh = tree2.query_ball_point(pos2[i], nn_cut)
        for p in neigh:
            j = idx2_img[p]
            if j == i:
                continue
            if subl2[i] != subl2[j]:
                if i < j:
                    ii = N1 + i
                    jj = N1 + j
                    rows += [ii, jj]
                    cols += [jj, ii]
                    data += [-gamma0, -gamma0]
    
    
    tree2_inter = cKDTree(pos2_img)
    for i in range(N1):
        neigh = tree2_inter.query_ball_point(pos1[i], interlayer_cut)
        for p in neigh:
            j = idx2_img[p]
            
            rxy = np.linalg.norm(pos1[i] - pos2_img[p])
            R = np.sqrt(rxy*rxy + d*d)
            t = t0 * np.exp(-(R - d)/r0)

            ii = i
            jj = N1 + j
            rows += [ii, jj]
            cols += [jj, ii]
            data += [t, t]
    
    H = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return H

def low_energy_bandwidth(H, k_eigs=12, take=6, sigma = 1e-6):
    
    H = (H + H.getH()) * 0.5
    evals = eigsh(H, k =k_eigs, sigma = sigma, which='LM', return_eigenvectors=False)
    evals = np.sort(np.real(evals))
    closest = evals[np.argsort(np.abs(evals))[:take]]
    mid = len(evals) // 2
    low = evals[mid-4:mid+4]
    W = low.max() - low.min()

    return W, evals


def reciprocal_vectors(t1, t2):
    A = np.column_stack((t1,t2))
    B = 2*np.pi * np.linalg.inv(A).T
    b1 = B[:, 0]
    b2 = B[:, 1]
    return b1, b2

def high_symmetry_points_moire(t1, t2):
    b1, b2 = reciprocal_vectors(t1, t2)
    G = np.array([0.0, 0.0])
    M = (b1 + b2) / 2.0
    K = (2 * b1 + b2) / 3.0
    return G, M, K

def make_k_path(points, n_per_seg):
    ks = []
    x = [0.0]
    dist = 0.0
    for A, B in zip(points[:-1], points[1:]):
        seg = np.linspace(0,1, n_per_seg, endpoint=False)
        for s in seg:
            k = (1-s) * A + s*B
            if ks:
                dist += np.linalg.norm(k - ks[-1])
                x.append(dist)
            ks.append(k)
    
    dist += np.linalg.norm(points[-1] - ks[-1])
    ks.append(points[-1])
    x.append(dist)
    return np.array(ks), np.array(x)


def build_sparse_H_tbg_k(pos1, subl1, pos2, subl2, t1, t2, kvec):
    N1 = len(pos1)
    N2 = len(pos2)
    N = N1 + N2

    rows, cols, data = [], [], []

    shifts = []
    shift_uv = []
    for u in [-1, 0, 1]:
        for v in [-1, 0, 1]:
            shifts.append(u*t1 + v*t2)
            shift_uv.append((u, v))
    shifts = np.array(shifts)
    shift_uv = np.array(shift_uv)

    
    pos1_img = (pos1[None,:,:] + shifts[:,None,:]).reshape(-1, 2)
    
    idx1_img = np.tile(np.arange(N1), len(shifts))
    
    sid1 = np.repeat(np.arange(len(shifts)), N1)

    tree1 = cKDTree(pos1_img)
    for i in range(N1):
        neigh = tree1.query_ball_point(pos1[i], nn_cut)
        for p in neigh:
            j = idx1_img[p]
            if j == i:
                continue
            if subl1[i] != subl1[j]:
                
                S = shifts[sid1[p]]
                phase = np.exp(1j * (kvec @ S))
                
                rows += [i, j]
                cols += [j, i]
                data += [-gamma0*phase, -gamma0*np.conjugate(phase)]

    
    pos2_img = (pos2[None,:,:] + shifts[:,None,:]).reshape(-1, 2)
    idx2_img = np.tile(np.arange(N2), len(shifts))
    sid2 = np.repeat(np.arange(len(shifts)), N2)

    tree2 = cKDTree(pos2_img)
    for i in range(N2):
        neigh = tree2.query_ball_point(pos2[i], nn_cut)
        for p in neigh:
            j = idx2_img[p]
            if j == i:
                continue
            if subl2[i] != subl2[j]:
                S = shifts[sid2[p]]
                phase = np.exp(1j * (kvec @ S))
                ii = N1 + i
                jj = N1 + j
                rows += [ii, jj]
                cols += [jj, ii]
                data += [-gamma0*phase, -gamma0*np.conjugate(phase)]

    
    tree2_inter = cKDTree(pos2_img)
    for i in range(N1):
        neigh = tree2_inter.query_ball_point(pos1[i], interlayer_cut)
        for p in neigh:
            j = idx2_img[p]
            S = shifts[sid2[p]]
            phase = np.exp(1j * (kvec @ S))

            rxy = np.linalg.norm(pos1[i] - pos2_img[p])
            R = np.sqrt(rxy*rxy + d*d)
            t = t0 * np.exp(-(R - d)/r0)

            ii = i
            jj = N1 + j
            rows += [ii, jj]
            cols += [jj, ii]
            data += [t*phase, t*np.conjugate(phase)]

    Hk = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    Hk = (Hk + Hk.getH()) * 0.5
    return Hk


def dos_from_kmesh(pos1, subl1, pos2, subl2, t1,t2, Nk=17, k_eigs = 40, sigma_e = 0.008, Emin = -0.15, Emax=0.15, nE=1200, sigma_shift=1e-4):

    b1, b2 = reciprocal_vectors(t1, t2)

    us = np.linspace(0, 1, Nk, endpoint=False)
    vs = np.linspace(0, 1, Nk, endpoint=False)

    evals_all = []

    for u in us:
        for v in vs:
            kvec = u*b1 + v*b2
            Hk = build_sparse_H_tbg_k(pos1, subl1, pos2, subl2, t1, t2, kvec)
            
            ev = eigsh(Hk, k=k_eigs, sigma=sigma_shift, which="LM",
                        return_eigenvectors=False, maxiter=5000)
            evals_all.append(np.real(ev))

    evals_all = np.concatenate(evals_all)

    E = np.linspace(Emin, Emax, nE)
    dos = np.zeros_like(E)
    pref = 1.0 / ((np.sqrt(2*np.pi) * sigma_e))  # normalization prefactor
    
    for e0 in evals_all:
        dos += pref * np.exp(-0.5 * ((E - e0)/sigma_e)**2)
    
    dos /= (Nk**2)  # average over k-points

    return E, dos, evals_all

import matplotlib.pyplot as plt

def run_bandwidth_vs_angle(mn_list=None):
    if mn_list is None:
        mn_list = [(5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15)]

    thetas_deg = []
    Ws = []

    for (m, n) in mn_list:
        theta = theta_from_mn(m, n)
        theta_deg = np.degrees(theta)

        pos1, subl1, t1, t2 = generate_layer1_in_cell(m, n)
        pos2 = wrap_to_cell(pos1 @ rot(theta).T, t1, t2)
        subl2 = subl1.copy()

        H = build_sparse_H_tbg(pos1, subl1, pos2, subl2, t1, t2)
        H = (H + H.getH()) * 0.5

        deg = np.diff(H.tocsr().indptr)
        #print("min nnz per row:", deg.min())
        #print("isolated rows (nnz==0):", np.sum(deg == 0))
        #print("very weak rows (nnz<=2):", np.sum(deg <= 2))

        W, _ = low_energy_bandwidth(H, k_eigs=14, take=8)
        #print(f"(m,n)=({m},{n})  θ={theta_deg:.3f}°  atoms={H.shape[0]}  W={W:.4f} eV")

        thetas_deg.append(theta_deg)
        Ws.append(W)

    thetas_deg = np.array(thetas_deg)
    Ws = np.array(Ws)

    order = np.argsort(thetas_deg)
    thetas_deg = thetas_deg[order]
    Ws = Ws[order]

    log_theta = np.log(thetas_deg)
    log_W = np.log(Ws)

    coeffs, cov = np.polyfit(log_theta, log_W, 1, cov=True)
    alpha = coeffs[0]
    A = np.exp(coeffs[1])

    slope_error = np.sqrt(cov[0,0])

    print("alpha: ", alpha)
    print("Slope Error: ", slope_error)

    theta_fit = np.linspace(thetas_deg.min(),thetas_deg.max(), 200)
    W_fit = A * theta_fit**alpha

    
    plt.figure(figsize=(7, 4))
    plt.loglog(thetas_deg, Ws, 'x', color='black', label='Data')
    plt.loglog(theta_fit, W_fit, label=f'Fit: α = {alpha:.2f}')
    plt.legend()
    
    plt.xlabel("Twist angle θ (degrees)")
    plt.ylabel("W (eV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TBG_bandwidth_vs_angle.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

def run_bandstructure_comparison(angles_to_compare=None):
    if angles_to_compare is None:
        angles_to_compare = [(5,6), (8,9), (14,15)]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=False)

    for ax, (m, n) in zip(axes, angles_to_compare):
        theta = theta_from_mn(m, n)
        theta_deg = np.degrees(theta)

        pos1, subl1, t1, t2 = generate_layer1_in_cell(m, n)
        pos2 = wrap_to_cell((pos1 @ rot(theta).T), t1, t2)
        subl2 = subl1.copy()

        G, K, M = high_symmetry_points_moire(t1, t2)
        ks, x = make_k_path([G, K, M, G], n_per_seg=25)

        bands = []
        for kvec in ks:
            Hk = build_sparse_H_tbg_k(pos1, subl1, pos2, subl2, t1, t2, kvec)
            Hk = (Hk + Hk.getH()) * 0.5
            evals = eigsh(Hk, k=14, sigma=1e-4, which="LM", return_eigenvectors=False)
            evals = np.sort(np.real(evals))
            mid = len(evals)//2
            bands.append(evals[mid-4:mid+4])

        bands = np.array(bands)
        for b in range(bands.shape[1]):
            ax.plot(x, bands[:, b], lw=1)

        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"θ = {theta_deg:.2f}°")
        ax.axhline(0, alpha=0.3)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xticks([x[0], x[len(x)//3], x[2*len(x)//3], x[-1]])
    axes[-1].set_xticklabels([r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    axes[-1].set_xlabel("k-path in moiré Brillouin zone")

    plt.tight_layout()
    plt.savefig("TBG_bandstructure_comparison.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

def run_dos(dos_angles=None):
    if dos_angles is None:
        dos_angles = [(5,6), (8,9), (10,11), (14,15)]

    theta_list, Evhs_list, Dvhs_list = [], [], []

    plt.figure(figsize=(7, 4))
    for (m, n) in dos_angles:
        theta = theta_from_mn(m, n)
        theta_deg = np.degrees(theta)

        pos1, subl1, t1, t2 = generate_layer1_in_cell(m, n)
        pos2 = wrap_to_cell(pos1 @ rot(theta).T, t1, t2)
        subl2 = subl1.copy()

        E, dos, _ = dos_from_kmesh(pos1, subl1, pos2, subl2, t1, t2,
                                   Nk=17, k_eigs=40, sigma_e=0.008,
                                   Emin=-0.15, Emax=0.15, nE=1200)

        plt.plot(E, dos, lw=1.5, label=f"{theta_deg:.2f}°")


    plt.axvline(0, alpha=0.3)
    plt.xlabel("Energy E (eV)")
    plt.ylabel("DOS (arb. units)")
    plt.title("DOS near charge neutrality vs twist angle")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("TBG_DOS_comparison.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


run_bandwidth_vs_angle()
#run_bandstructure_comparison()
#run_dos()