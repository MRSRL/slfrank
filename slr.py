import numpy as np
import cvxpy as cp
import scipy


def rotate(a, b, b1, dt, gamma):
    """Apply hard pulse rotation to input magnetization.

    Args:
        theta (complex float): complex B1 value in radian.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    phi = np.abs(b1) * gamma * dt
    theta = np.angle(b1)

    c = np.cos(phi / 2)
    s = 1j * np.exp(1j * theta) * np.sin(phi / 2)
    return c * a - s.conjugate() * b, s * a + c * b


def inverse_rotate(a, b, dt, gamma):
    """
    """
    phi = 2 * np.arctan2(np.abs(b[0]), np.abs(a[0]))
    theta = np.angle(-1j * b[0] * a[0].conjugate())
    b1 = phi * np.exp(1j * theta) / (gamma * dt)

    c = np.cos(phi / 2)
    s = 1j * np.exp(1j * theta) * np.sin(phi / 2)
    return c * a + s.conjugate() * b, -s * a + c * b, b1


def precess(a, b):
    """Simulate precess to input magnetization.

    Args:
        p (array): magnetization array.
        omega (array): off-resonance.
        dt (float): free induction decay duration.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    a = np.concatenate((a, [0]))
    b = np.concatenate(([0], b))
    return a, b


def inverse_precess(a, b):
    """Simulate precess to input magnetization.

    Args:
        p (array): magnetization array.
        omega (array): off-resonance.
        dt (float): free induction decay duration.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    return a[:-1], b[1:]


def forward_slr(b1, dt, gamma=26752.218744):
    """Shinnar Le Roux forward evolution.

    The function uses the hard pulse approximation. Given an array of
    B1 complex amplitudes, it simulates a sequence of precess
    followed by a hard pulse rotation.

    Args:
        b1 (array): complex B1 array in Gauss.
        dt (float): delta time in seconds.
        gamma (float): gyromagnetic ratio in radian / seconds / Gauss.

    Returns:
        array: polynomial of shape (n, 2)

    Examples:
        Simulating an on-resonant spin under 90 degree pulse.
        The 90 degree pulse is discretized into 1000 time points.
        1 ms.

        >>> gamma = 26752.218744  # radian / seconds / Gauss
        >>> dt = 1e-6
        >>> n = 1000
        >>> b1 = np.pi / 2 / (dt * n) / gamma * np.ones(n)
        >>> a, b = forward_slr(b1, dt)

    """
    a, b = rotate(np.array([1]), np.array([0]), b1[0], dt, gamma)
    for b1_i in b1[1:]:
        a, b = precess(a, b)
        a, b = rotate(a, b, b1_i, dt, gamma)

    return a, b


def eval_slr(a, b, omega):
    """Evaluate Shinnar-Le-roux polynomials at given frequencies.

    Args:
        a (array): alpha polynomial of length n.
        b (array): beta polynomial of length n.
        omega (float or array): frequencies in radian.
            Can be a scalar or an array of length m.

    Returns:
        array, array: transverse and longitudinal magnetization
            of length m.

    """
    n = len(a)
    psi_z = np.exp(-1j * np.outer(np.arange(n), omega))
    a = a @ psi_z
    b = b @ psi_z
    m_xy = 2 * a.conjugate() * b
    m_z = np.abs(a)**2 - np.abs(b)**2
    return m_xy, m_z


def inverse_slr(a, b, dt, gamma=26752.218744):
    """Shinnar Le Roux inverse evolution.

    The function uses the hard pulse approximation. Given an array of
    B1 complex amplitudes, it simulates a sequence of precess
    followed by a hard pulse rotation.

    Args:
        a (array): alpha polynomial of length n.
        b (array): beta polynomial of length n.
        dt (float): delta time in seconds.
        gamma (float): gyromagnetic ratio in radian / seconds / Gauss.

    Returns:
        array: polynomial of shape (n)

    """
    n = len(a)
    b1 = []
    for i in range(n):
        a, b, b1_i = inverse_rotate(a, b, dt, gamma)
        a, b = inverse_precess(a, b)
        b1 = [b1_i] + b1

    return np.array(b1)


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


def bound_constraints(x, bands, vals, delta, oversample=10, phase_shift=0):
    n = (x.shape[0] - 1) // 2
    m = n * oversample
    omega = np.random.default_rng().uniform(-np.pi, np.pi, m)
    A = np.exp(-1j * np.outer(omega, np.arange(-n, n + 1)))
    i = np.zeros(m, dtype=bool)
    y = np.zeros(m, dtype=np.complex)

    for band, val in zip(bands, vals):
        idx = (omega >= band[0]) & (omega <= band[1])
        y[idx] = val * np.exp(-1j * omega[idx] * phase_shift)
        i |= idx

    constraints = [cp.abs(A[i] @ x - y[i]) <= delta]
    return constraints


def design_linear_phase_slr(n, bands, flip_angles, delta, verbose=False, tol=1e-3):
    """
    Equation 2.45 page 36
    """
    X = cp.Variable((2 * n, 2 * n), complex=True)

    aa = X[:n, :n]
    ba = X[n:, :n]
    bb = X[n:, n:]
    m_xy = 2 * diag_sum(ba)

    dirac = np.zeros(2 * n - 1, dtype=np.complex)
    dirac[n - 1] = 1

    constraints = [X >> 0, diag_sum(aa) + diag_sum(bb) == dirac]
    constraints += bound_constraints(
        m_xy, bands, np.sin(flip_angles), delta, phase_shift=n / 2)

    objective = cp.Maximize(cp.real(aa[0, 0]))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, solver=cp.MOSEK)

    if prob.status == 'infeasible':
        raise ValueError('Infeasible: try relaxing constraints.')

    w, v = np.linalg.eigh(X.value)
    a = v[:n, -1]
    b = v[n:, -1]
    a *= w[-1]**0.5
    b *= w[-1]**0.5

    if w[-2] / w[-1] > tol:
        raise ValueError(f'Convex relaxation not right, got lambda_2 / lambda_1 ={w[-2] / w[-1]}.')

    return a, b
