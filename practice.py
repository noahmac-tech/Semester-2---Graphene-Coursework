import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh

a = 2.46
gamma0 = 3


"""Define grid in reciprocal space"""
K = (4*np.pi)/(3*a)
N = 400

kx = np.linspace(-K,K,N)
ky = np.linspace(-K,K,N)

KX, KY = np.meshgrid(kx, ky)


"""Calculate positive and negative surfaces"""
"""Plot on same 3D axes"""
