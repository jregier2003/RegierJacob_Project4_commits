import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    dx = length / nspace
    x = np.linspace(-length / 2, length / 2, nspace)
    t = np.arange(0, ntime * tau, tau)

    sigma0, x0, k0 = wparam
    psi = np.zeros((ntime, nspace), dtype=complex)
    psi[0, :] = np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)
    psi[0, :] /= np.sqrt(np.sum(np.abs(psi[0, :])**2) * dx)  

    V = np.zeros(nspace)
    for idx in potential:
        if 0 <= idx < nspace:
            V[idx] = 1

    if method == 'ftcs':
        alpha = tau / dx**2
        for n in range(ntime - 1):
            for i in range(1, nspace - 1):
                psi[n + 1, i] = psi[n, i] - 1j * alpha * (psi[n, i + 1] - 2 * psi[n, i] + psi[n, i - 1]) + 1j * tau * V[i] * psi[n, i]
            psi[n + 1, 0] = psi[n + 1, -2] 
            psi[n + 1, -1] = psi[n + 1, 1] 

    return psi, x, t, V


def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    matrix = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix


def compute_spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))


def adjust_alpha_for_stability(alpha, nspace):
    max_attempts = 100
    adjustment_factor = 0.9
    for _ in range(max_attempts):
        evolution_matrix = create_tridiagonal_matrix(nspace, -alpha, 1 + 2 * alpha, -alpha)
        radius = compute_spectral_radius(evolution_matrix)
        if radius <= 1:
            print(f"Adjusted alpha to: {alpha} (Spectral Radius: {radius})")
            return alpha
        alpha *= adjustment_factor
    print("Warning: Could not stabilize FTCS.")
    return alpha

