import numpy as np
import cvxpy as cp
import sigpy as sp
import sigpy.mri.rf as rf
import scipy.sparse
from . import linop, prox, transform


def design_rf(n=64, tb=4, ptype='ex', d1=0.01, d2=0.01, phase='linear',
              oversamp=8, lamda=None, solver='PDHG',
              max_iter=None, sigma=None, verbose=True):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    bands = [[-np.pi, -(1 + w) * tb / n * np.pi],
             [-(1 - w) * tb / n * np.pi, (1 - w) * tb / n * np.pi],
             [(1 + w) * tb / n * np.pi, np.pi]]

    if ptype == 'ex' and phase == 'linear':
        m_xy_vals = [0, lambda omega: np.exp(-1j * omega * n / 2), 0]
        m_xy_deltas = [d2, d1, d2]
        m_z_vals = [1, 0, 1]
        m_z_deltas = [1 - (1 - d2**2)**0.5, (1 - (1 - d1)**2)**0.5, 1 - (1 - d2**2)**0.5]
        beta_vals = []
        beta_deltas = []
        if lamda is None:
            lamda = 0
        if solver == 'PDHG':
            if sigma is None:
                sigma = 1000
            if max_iter is None:
                max_iter = 3000
    elif ptype == 'ex' and phase == 'min':
        m_xy_vals = [0, 0, 0]
        m_xy_deltas = [d2, 1, d2]
        m_z_vals = [1, 0, 1]
        m_z_deltas = [1 - (1 - d2**2)**0.5, (1 - (1 - d1)**2)**0.5, 1 - (1 - d2**2)**0.5]
        beta_vals = []
        beta_deltas = []
        if lamda is None:
            lamda = 1
        if sigma is None:
            sigma = 100
        if max_iter is None:
            max_iter = 20000
    elif ptype == 'sat' and phase == 'min':
        m_xy_vals = [0, 0, 0]
        m_xy_deltas = [(1 - (1 - d2)**2)**0.5, 1, (1 - (1 - d2)**2)**0.5]
        m_z_vals = [1, 0, 1]
        m_z_deltas = [d2, d1, d2]
        beta_vals = []
        beta_deltas = []
        if lamda is None:
            lamda = 1
        if sigma is None:
            sigma = 1000
        if max_iter is None:
            max_iter = 10000
    elif ptype == 'inv' and phase == 'min':
        m_xy_vals = [0, 0, 0]
        m_xy_deltas = [(1 - (1 - d2)**2)**0.5, (1 - (1 - d1)**2)**0.5, (1 - (1 - d2)**2)**0.5]
        m_z_vals = [1, -1, 1]
        m_z_deltas = [d2, d1, d2]
        beta_vals = []
        beta_deltas = []
        if lamda is None:
            lamda = 1
        if solver == 'PDHG':
            if sigma is None:
                sigma = 1000
            if max_iter is None:
                max_iter = 20000
    elif ptype == 'se' and phase == 'linear':
        m_xy_vals = []
        m_xy_deltas = []
        m_z_vals = []
        m_z_deltas = []
        beta_vals = [0, lambda omega: np.exp(-1j * omega * (n - 1) / 2), 0]
        beta_deltas = [d2**0.5, (1 - (1 - d1)**0.5) / 2, d2**0.5]
        if lamda is None:
            lamda = 0
        if solver == 'PDHG':
            if sigma is None:
                sigma = 1000
            if max_iter is None:
                max_iter = 3000
    else:
        raise ValueError(f'ptype={ptype} and phase={phase} not implemented.')

    m = n * oversamp
    omega = 2 * np.pi * (np.arange(m) - m // 2) / m
    m_xy, d_xy = bands_to_arrays(omega, bands, m_xy_vals, m_xy_deltas)
    m_z, d_z = bands_to_arrays(omega, bands, m_z_vals, m_z_deltas)
    beta, d_beta = bands_to_arrays(omega, bands, beta_vals, beta_deltas)

    if solver == 'PDHG':
        a, b = DesignPaulynomials(
            n, omega, m_xy, d_xy, m_z, d_z, beta, d_beta,
            max_iter=max_iter, sigma=sigma, lamda=lamda).run()
    else:
        a, b = design_paulynomials(
            n, omega, m_xy, d_xy, m_z, d_z, beta, d_beta,
            lamda=lamda, solver=solver, verbose=verbose)

    b1 = transform.inverse_slr(a, b * 1j)
    return b1


def bands_to_arrays(omega, bands, vals, deltas):
    """Convert M_xy and M_z band specifications to arrays.

    bands (list of bands): list of frequency bands, specified by
         starting and end points in radians between -pi to pi.
    vals (list of floats): desired magnitude response for each band.
        Must have the same length as bands. For example,
        [0, 1, 0].
    linphases (list of floats): desired linear phase for each band.
        Must have the same length as bands.
    deltas (list of floats): maximum deviation from specified profile.
    """
    m = len(omega)
    v = np.zeros(m, dtype=np.complex)
    d = np.ones(m)

    for band, val, delta, in zip(bands, vals, deltas):
        i = (omega >= band[0]) & (omega <= band[1])
        if np.isscalar(val):
            v[i] = val
        else:
            v[i] = val(omega[i])

        if np.isscalar(delta):
            d[i] = delta
        else:
            d[i] = delta(omega[i])

    return v, d


class DesignPaulynomials(sp.app.App):
    """Design Shinnar-Le-Roux polynomials given magnetization profiles.

    Args:
        n (int): number of hard pulses.
        m_xy (array): transverse magnetization.
        sigma (float): dual step-size.
        max_iter (int): maximum number of iterations.
        m (int): number of points for discretizing the frequency response.

    Example:
        n = 64
        bands = [[-np.pi, -0.5 * np.pi],
                 [-0.25 * np.pi, 0.25 * np.pi],
                 [0.5 * np.pi, np.pi]]
        vals = [0, 1, 0]
        linphases = [0, n / 2, 0]
        deltas = [0.01, 0.01, 0.01]

    Returns:
        array, array: alpha and beta polynomials of length n.

    """
    def __init__(self, n, omega, m_xy, d_xy, m_z, d_z, beta, d_beta,
                 sigma=100, lamda=0, max_iter=10000):

        m = len(m_xy)
        # Create linear operators
        A_11 = sp.linop.Slice((2 * n + 1, 2 * n + 1), (slice(1), slice(1)))
        S_b = sp.linop.Slice((2 * n + 1, 2 * n + 1), (slice(n + 1, 2 * n + 1), 0))
        S_aa = sp.linop.Slice((2 * n + 1, 2 * n + 1), (slice(1, n + 1), slice(1, n + 1)))
        S_ba = sp.linop.Slice((2 * n + 1, 2 * n + 1), (slice(n + 1, 2 * n + 1), slice(1, n + 1)))
        S_bb = sp.linop.Slice(
            (2 * n + 1, 2 * n + 1), (slice(n + 1, 2 * n + 1), slice(n + 1, 2 * n + 1)))

        D = linop.DiagSum(n)
        F = np.exp(-1j * np.outer(omega, np.arange(-n + 1, n)))        
        F_b = np.exp(-1j * np.outer(omega, np.arange(n)))
        A_b = sp.linop.Reshape((m, ), (m, 1)) * sp.linop.MatMul((n, 1), F_b) * sp.linop.Reshape((n, 1), (n, )) * S_b

        A_xy = sp.linop.Reshape((m, ), (m, 1)) * sp.linop.MatMul((2 * n - 1, 1), F) * sp.linop.Reshape((2 * n - 1, 1), (2 * n - 1, )) * D * (2 * S_ba)
        A_z = sp.linop.Reshape((m, ), (m, 1)) * sp.linop.MatMul((2 * n - 1, 1), F) * sp.linop.Reshape((2 * n - 1, 1), (2 * n - 1, )) * D * (S_aa - S_bb)
        A_I = D * (S_aa + S_bb)
        As = [A_11, A_b, A_xy, A_z, A_I]
        A = sp.linop.Vstack(As)

        # Create proximal operators
        dirac = np.zeros(2 * n - 1)
        dirac[n - 1] = 1

        proxf_1 = sp.prox.LInfProj([1], 0, 1)
        proxf_b = sp.prox.LInfProj(A_b.oshape, d_beta, beta)
        proxf_xy = sp.prox.LInfProj(A_xy.oshape, d_xy, m_xy)
        proxf_z = sp.prox.LInfProj(A_z.oshape, d_z, m_z)
        proxf_I = sp.prox.LInfProj(A_I.oshape, 0, dirac)
        proxf = sp.prox.Stack([
            proxf_1, proxf_b, proxf_xy, proxf_z, proxf_I])
        proxfc = sp.prox.Conj(proxf)

        proxg = prox.Objective((2 * n + 1, 2 * n + 1), lamda)

        # Get step-size
        sigma_1 = sigma
        sigma_b = sigma / m
        sigma_xy = sigma / (4 * n * m)
        sigma_z = sigma / (2 * n * m)
        sigma_I = sigma / (2 * n)
        tau = 1 / sigma
        sigma = np.concatenate(
            [[sigma_1],
             np.full(A_b.oshape, sigma_b),
             np.full(A_xy.oshape, sigma_xy),
             np.full(A_z.oshape, sigma_z),
             np.full(A_I.oshape, sigma_I)])

        # Get variables
        self.X = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex)
        self.X[0, 0] = 1
        self.a = self.X[1:n + 1, 0]
        self.b = self.X[n + 1:, 0]
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.X, u, tau, sigma,
            max_iter=max_iter)
        self.m_xy = m_xy
        self.m_z = m_z
        self.beta = beta
        self.d_xy = d_xy
        self.d_z = d_z
        self.d_beta = d_beta
        self.As = As
        self.lamda = lamda
        self.dirac = dirac
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            A_11, A_b, A_xy, A_z, A_I = self.As
            err_beta = np.max(
                np.clip(np.abs(A_b(self.X) - self.beta) - self.d_beta, 0, None))
            err_xy = np.max(
                np.clip(np.abs(A_xy(self.X) - self.m_xy) - self.d_xy, 0, None))
            err_z = np.max(
                np.clip(np.abs(A_z(self.X) - self.m_z) - self.d_z, 0, None))
            err_I = np.max(np.abs(A_I(self.X) - self.dirac))

            w, v = np.linalg.eigh(self.X)
            err_rank1 = np.sum(w[:-1])
            n = (len(self.X) + 1) // 2
            obj = self.X[1, 0]
            obj += self.X[0, 1]
            obj += self.lamda * self.X[0, n + 1]
            obj += self.lamda * self.X[n + 1, 0]
            obj = np.real(obj)
            self.pbar.set_postfix(err_rank1='{0:.2E}'.format(err_rank1),
                                  err_beta='{0:.2E}'.format(err_beta),
                                  err_I='{0:.2E}'.format(err_I),
                                  err_xy='{0:.2E}'.format(err_xy),
                                  err_z='{0:.2E}'.format(err_z),
                                  obj='{0:.2E}'.format(obj))

    def _output(self):
        return self.a, self.b


def design_paulynomials(n, omega, m_xy, d_xy, m_z, d_z, beta, d_beta,
                        verbose=False, lamda=0, solver='SCS'):
    """Design Paulynomials.

    """
    # Initialize variables
    X = cp.Variable((2 * n + 1, 2 * n + 1), complex=True)
    X = (X + cp.conj(X.T)) / 2
    a = X[1:n + 1, 0]
    b = X[n + 1:, 0]
    P = X[1:, 1:]

    aa = P[:n, :n]
    ba = P[n:2 * n, :n]
    bb = P[n:2 * n, n:2 * n]

    # Get constraints
    constraints = [X >> 0, X[0, 0] == 1]

    # Trace equals to 1
    dirac = np.zeros(2 * n - 1, dtype=np.complex)
    dirac[n - 1] = 1
    constraints.append(diag_sum(aa + bb) == dirac)

    if m_xy is not None:
        A = np.exp(-1j * np.outer(omega, np.arange(-n + 1, n)))
        constraints.append(
            cp.abs(m_xy - 2 * A @ diag_sum(ba)) <= d_xy)
    if m_z is not None:
        A = np.exp(-1j * np.outer(omega, np.arange(-n + 1, n)))
        constraints.append(
            cp.abs(m_z - A @ diag_sum(aa - bb)) <= d_z)
    if beta is not None:
        A = np.exp(-1j * np.outer(omega, np.arange(n)))
        constraints.append(cp.abs(beta - A @ b) <= d_beta)

    # Set up problem
    objective = cp.Maximize(cp.real(a[0] + lamda * b[0]))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, solver=solver)

    if verbose:
        w, v = np.linalg.eigh(X.value)
        print(f'Rank-1 approximation error: {np.sum(w[:-1])}')

    if prob.status == 'infeasible':
        raise ValueError('Infeasible: try relaxing constraints.')

    return a.value, b.value


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
