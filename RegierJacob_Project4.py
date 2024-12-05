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
        alpha = adjust_alpha_for_stability(alpha, nspace)
        for n in range(ntime - 1):
            for i in range(1, nspace - 1):
                psi[n + 1, i] = psi[n, i] - 1j * alpha * (psi[n, i + 1] - 2 * psi[n, i] + psi[n, i - 1]) + 1j * tau * V[i] * psi[n, i]
            psi[n + 1, 0] = psi[n + 1, -2] 
            psi[n + 1, -1] = psi[n + 1, 1] 

    elif method == 'crank':
        alpha = 1j * tau / (2 * dx**2)
        beta = 1j * tau / 2
        A = create_tridiagonal_matrix(nspace, -alpha, 1 + 2 * alpha + beta * V, -alpha)
        B = create_tridiagonal_matrix(nspace, alpha, 1 - 2 * alpha - beta * V, alpha)
        for n in range(ntime - 1):
            rhs = B @ psi[n, :]
            psi[n + 1, :] = solve(A, rhs)

    prob = np.zeros(ntime)
    for n in range(ntime):
        prob[n] = np.sum(np.abs(psi[n, :])**2) * dx

    return psi, x, t, prob


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


def sch_plot(psi, x, t, plot_type='psi', time_index=0):
    plt.figure(figsize=(8, 5))
    if plot_type == 'psi':
        plt.plot(x, np.real(psi[time_index, :]), label='Real(ψ)', lw=2)
    elif plot_type == 'prob':
        plt.plot(x, np.abs(psi[time_index, :])**2, label='|ψ|²', lw=2)
    plt.title(f"{plot_type.capitalize()} at t={t[time_index]:.2f}")
    plt.xlabel("x")
    plt.ylabel("Amplitude" if plot_type == 'psi' else "Probability")
    plt.legend()
    plt.show()


def test_schrodinger():
    test_params = [
        {"tau": 0.0025, "nspace": 150},
        {"tau": 0.002, "nspace": 200},
        {"tau": 0.0015, "nspace": 250},
    ]
    for params in test_params:
        print(f"\nTesting with tau={params['tau']}, nspace={params['nspace']}:")
        try:
            psi_ftcs, x, t, prob_ftcs = sch_eqn(nspace=params['nspace'], ntime=500, tau=params['tau'], method='ftcs')
            print("FTCS Total Probability (Last Step):", prob_ftcs[-1])
            sch_plot(psi_ftcs, x, t, plot_type='prob', time_index=250)
        except ValueError as e:
            print("FTCS Test Failed:", e)
    psi_crank, x, t, prob_crank = sch_eqn(nspace=100, ntime=500, tau=0.01, method='crank')
    print("\nCrank-Nicholson Total Probability (Last Step):", prob_crank[-1])
    sch_plot(psi_crank, x, t, plot_type='prob', time_index=250)

test_schrodinger()