import numpy as np
import scipy.sparse
import cvxpy as cp


def diag_sum(X):
    n = X.shape[0] - 1
    si = np.zeros((n + 1)**2)
    sj = np.zeros((n + 1)**2)
    ss = np.ones((n + 1)**2)

    c = 0
    for i in range(n + 1):
        for j in range(n + 1):
            si[c] = i - j + n
            sj[c] = i + j * (n + 1)
            c = c + 1

    shape = (2 * n + 1, (n + 1)**2)
    A = scipy.sparse.coo_matrix((ss, (si, sj)), shape=shape)
    return A @ cp.vec(X)


def freq_constraint(x, bands, vals, deltas, oversamp, conj=False):
    n = (x.shape[0] - 1) // 2
    omega = np.linspace(-np.pi, np.pi, n * oversamp)
    A = np.exp(-1j * np.outer(omega, np.arange(-n, n + 1)))
    y = np.zeros(n * oversamp, dtype=np.complex)
    d = np.ones(n * oversamp)

    for band, val, delta in zip(bands, vals, deltas):
        idx = (omega >= band[0]) & (omega <= band[1])
        d[idx] = delta
        if np.isscalar(val):
            y[idx] = val
        else:
            y[idx] = val(omega[idx])

    if conj:
        y = np.conj(y)

    return cp.abs(y - A @ cp.vec(x)) <= d


def beta_freq_constraint(b, bands, vals, deltas, oversamp):
    n = b.shape[0]
    omega = np.linspace(-np.pi, np.pi, n * oversamp)
    A = np.exp(-1j * np.outer(omega, np.arange(n)))
    i = np.zeros(n * oversamp, dtype=bool)
    y = np.zeros(n * oversamp, dtype=np.complex)
    d = np.zeros(n * oversamp)

    for j in range(len(bands)):
        idx = (omega >= bands[j][0]) & (omega <= bands[j][1])
        i |= idx
        d[idx] = deltas[j]
        if np.isscalar(vals[j]):
            y[idx] = vals[j]
        else:
            y[idx] = vals[j](omega[idx])

    return cp.abs(y[i] - A[i] @ cp.vec(b)) <= d[i]


def design_paulynomials(n, bands,
                        m_xy=None, d_xy=None, m_z=None, d_z=None,
                        beta=None, d_beta=None,
                        verbose=False, oversamp=10, lamda=0):
    """Design Paulynomials.

    """
    # Initialize variables
    X = cp.Variable((2 * n + 1, 2 * n + 1), complex=True)
    a = X[1:n + 1, 0]
    b = X[n + 1:, 0]
    ac = X[0, 1:n + 1]
    bc = X[1:n + 1, 0]
    P = X[1:, 1:]

    aa = P[:n, :n]
    ba = P[n:2 * n, :n]
    ab = P[:n, n:2 * n]
    bb = P[n:2 * n, n:2 * n]

    # Get constraints
    constraints = [X >> 0, X[0, 0] == 1]

    # Trace equals to 1
    dirac = np.zeros(2 * n - 1, dtype=np.complex)
    dirac[n - 1] = 1
    constraints.append(diag_sum(aa + bb) == dirac)

    if m_xy is not None:
        constraints.append(
            freq_constraint(2 * diag_sum(ba), bands, m_xy, d_xy, oversamp))
        constraints.append(
            freq_constraint(
                2 * diag_sum(ab), bands, m_xy, d_xy, oversamp, conj=True))
    if m_z is not None:
        constraints.append(
            freq_constraint(diag_sum(aa - bb), bands, m_z, d_z, oversamp))
    if beta is not None:
        constraints.append(
            beta_freq_constraint(b, bands, beta, d_beta, oversamp))

    # Set up problem
    objective = cp.Maximize(cp.real(a[0] + ac[0])
                            + lamda * cp.real(b[0] + bc[0]))
    prob = cp.Problem(objective, constraints)
    if 'MOSEK' in cp.installed_solvers():
        solver = 'MOSEK'
    else:
        solver = 'SCS'

    prob.solve(verbose=verbose, solver=solver)

    if verbose:
        w, v = np.linalg.eigh(X.value)
        print(f'Rank-1 approximation error: {np.sum(w[:-1])}')

    if prob.status == 'infeasible':
        raise ValueError('Infeasible: try relaxing constraints.')

    return a.value, b.value
