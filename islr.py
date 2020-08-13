import numpy as np
import sigpy as sp
import linop
import sigpy.mri.rf as rf


def dzrf(n=64, tb=4, ptype='ex', d1=0.01, d2=0.01,
         m=1000, max_iter=10000, sigma=100, eps=None):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    bands = [[-np.pi, -(1 + w) * tb / n * np.pi],
             [-(1 - w) * tb / n * np.pi, (1 - w) * tb / n * np.pi],
             [(1 + w) * tb / n * np.pi, np.pi]]

    if ptype == 'ex':
        m_xy_vals = [0, lambda omega: np.exp(1j * omega * (n - 1) / 2), 0]
        delta_xys = [d2, d1, d2]
        m_z_vals = [1, 0, 1]
        delta_zs = [1 - (1 - d2**2)**0.5,
                    (1 - (1 - d1)**2)**0.5,
                    1 - (1 - d2**2)**0.5]
        if eps is None:
            eps = 0
    elif ptype == 'ex-minphase':
        m_xy_vals = [0, 0, 0]
        delta_xys = [d2, 1, d2]
        m_z_vals = [1, 0, 1]
        delta_zs = [1 - (1 - d2**2)**0.5,
                    (1 - (1 - d1)**2)**0.5,
                    1 - (1 - d2**2)**0.5]
        if eps is None:
            eps = 1
    elif ptype == 'inv':
        m_xy_vals = [0, 0, 0]
        delta_xys = [(1 - (1 - d2)**2)**0.5,
                     (1 - (1 - d1)**2)**0.5,
                     (1 - (1 - d2)**2)**0.5]
        m_z_vals = [1, -1, 1]
        delta_zs = [d2, d1, d2]
        if eps is None:
            eps = 1

    m_xy, m_z, d_xy, d_z = bands_to_arrays(
        m, bands, m_xy_vals, delta_xys, m_z_vals, delta_zs)

    a, b = DesignPaulynomials(n, m_xy, m_z, d_xy, d_z,
                              max_iter=max_iter, sigma=sigma, eps=eps).run()
    b1 = rf.ab2rf(a, b)
    return b1


def dzmbrf(n=256, tb=4, ptype='ex', d1=0.01, d2=0.01,
           n_bands=3, band_sep=12, phs_0_pt='None',
           m=1000, max_iter=10000, sigma=100, eps=0):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    if phs_0_pt != 'None':
        phs = rf.multiband.mb_phs_tab(n_bands, phs_0_pt)
    else:
        phs = np.zeros(n_bands)

    def get_m_xy_func(i):
        return lambda omega: np.exp(1j * phs[i] + 1j * omega * (n - 1) / 2)

    bands = []
    m_xy_vals = []
    delta_xys = []
    m_z_vals = []
    delta_zs = []
    # First stop band
    center = 2 * np.pi * band_sep * (- (n_bands - 1) / 2) / n
    bands.append([-np.pi, center - (1 + w) * tb / n * np.pi])
    m_xy_vals.append(0)
    delta_xys.append(d2)
    m_z_vals.append(1)
    delta_zs.append(1 - (1 - d2**2)**0.5)

    for i in range(n_bands):
        center = 2 * np.pi * band_sep * (i - (n_bands - 1) / 2) / n
        # Pass band
        bands.append([center - (1 - w) * tb / n * np.pi,
                      center + (1 - w) * tb / n * np.pi])
        m_xy_vals.append(get_m_xy_func(i))
        delta_xys.append(d1)
        m_z_vals.append(0)
        delta_zs.append((1 - (1 - d1)**2)**0.5)

        # Stop band
        if i == n_bands - 1:
            end = np.pi
        else:
            next_center = 2 * np.pi * band_sep * (i + 1 - (n_bands - 1) / 2) / n
            end = next_center - (1 + w) * tb / n * np.pi

        bands.append([center + (1 + w) * tb / n * np.pi, end])
        m_xy_vals.append(0)
        delta_xys.append(d2)
        m_z_vals.append(1)
        delta_zs.append(1 - (1 - d2**2)**0.5)

    m_xy, m_z, d_xy, d_z = bands_to_arrays(
        m, bands, m_xy_vals, delta_xys, m_z_vals, delta_zs)

    a, b = DesignPaulynomials(n, m_xy, m_z, d_xy, d_z,
                              max_iter=max_iter, sigma=sigma, eps=eps).run()
    b1 = rf.ab2rf(a, b)
    return b1


def bands_to_arrays(m, bands, m_xy_vals, delta_xys,
                    m_z_vals, delta_zs):
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
    omega = 2 * np.pi * (np.arange(m) - m // 2) / m
    m_xy = np.zeros(m, dtype=np.complex)
    m_z = np.zeros(m, dtype=np.complex)
    d_xy = np.full(m, 1e10)
    d_z = np.full(m, 1e10)

    for band, m_xy_val, delta_xy, m_z_val, delta_z, in zip(
            bands, m_xy_vals, delta_xys, m_z_vals, delta_zs):
        i = (omega >= band[0]) & (omega <= band[1])
        if np.isscalar(m_xy_val):
            m_xy[i] = m_xy_val
        else:
            m_xy[i] = m_xy_val(omega[i])

        if np.isscalar(m_z_val):
            m_z[i] = m_z_val
        else:
            m_z[i] = m_z_val(omega[i])

        if np.isscalar(delta_xy):
            d_xy[i] = delta_xy
        else:
            d_xy[i] = delta_xy(omega[i])

        if np.isscalar(delta_z):
            d_z[i] = delta_z
        else:
            d_z[i] = delta_z(omega[i])

    return m_xy, m_z, d_xy, d_z


def P_to_ab(P):
    w, v = np.linalg.eigh(P)
    x = v[:, -1] * w[-1]**0.5

    n = P.shape[0] // 2
    a = x[:n] * np.sign(x[0])
    b = x[n:] * np.sign(x[0])
    return a, b


class DesignPaulynomials(sp.app.App):
    """Design Shinnar-Le-Roux polynomials given an Mxy profile.

    Args:
        n (int): number of hard pulses.
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
    def __init__(self, n, m_xy, m_z, d_xy, d_z,
                 sigma=100, eps=0, max_iter=10000):

        m = len(m_xy)
        # Create linear operators
        S_aa = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n)))
        S_ba = sp.linop.Slice((2 * n, 2 * n), (slice(n, 2 * n), slice(n)))
        S_ab = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n, 2 * n)))
        S_bb = sp.linop.Slice(
            (2 * n, 2 * n), (slice(n, 2 * n), slice(n, 2 * n)))

        D = linop.DiagSum(n)
        R = sp.linop.Resize((m, ), D.oshape)
        F = sp.linop.FFT((m, ))

        A_xy = m**0.5 * F * R * D * (2 * S_ba)
        A_xyc = m**0.5 * F * R * D * (2 * S_ab)
        A_z = m**0.5 * F * R * D * (S_aa - S_bb)
        A_I = D * (S_aa + S_bb)
        A_psd = sp.linop.Identity((2 * n, 2 * n))
        As = [A_xy, A_xyc, A_z, A_I, A_psd]
        A = sp.linop.Vstack(As)

        # Create proximal operators
        dirac = np.zeros(2 * n - 1)
        dirac[n - 1] = 1

        def proxfc(alpha, u):
            u_xy, u_xyc, u_z, u_I, u_psd = np.split(
                u, [m, 2 * m, 3 * m, 3 * m + 2 * n - 1])
            alpha_xy, alpha_xyc, alpha_z, alpha_I, alpha_psd = np.split(
                alpha, [m, 2 * m, 3 * m, 3 * m + 2 * n - 1])

            u_xy[:] = sp.soft_thresh(alpha_xy * d_xy, u_xy - alpha_xy * m_xy)
            u_xyc[:] = sp.soft_thresh(alpha_xyc * d_xy, u_xyc - alpha_xyc * m_xy.conjugate())
            u_z[:] = sp.soft_thresh(alpha_z * d_z, u_z - alpha_z * m_z)
            u_I -= alpha_I * dirac
            u_psd = u_psd.reshape((2 * n, 2 * n))
            w, v = np.linalg.eigh(u_psd)
            w[w > 0] = 0
            u_psd[:] = (v * w) @ v.conjugate().T

            return u

        def proxg(alpha, P):
            P[n - 1, n - 1] += alpha
            P[n - 1, 2 * n - 1] += alpha * eps
            P[2 * n - 1, n - 1] += alpha * eps
            return P

        # Get step-size
        Ls = [sp.app.MaxEig(A_i.H * A_i, dtype=np.complex, show_pbar=False).run() for A_i in As]
        sigmas = [np.full(sp.prod(A_i.oshape), sigma / L_i)
                  for L_i, A_i in zip(Ls, As)]
        sigma = np.concatenate(sigmas)
        S = sp.linop.Multiply(A.oshape, sigma)
        L = sp.app.MaxEig(A.H * S * A, dtype=np.complex, show_pbar=False).run()
        tau = 1 / L

        self.P = np.zeros(A.ishape, dtype=np.complex)
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.P, u, tau, sigma, max_iter=max_iter)
        self.m_xy = m_xy
        self.m_z = m_z
        self.d_xy = d_xy
        self.d_z = d_z
        self.As = As
        self.eps = eps
        self.dirac = dirac
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            A_xy, A_xyc, A_z, A_I, A_psd = self.As
            err_xy = np.abs(A_xy(self.P) - self.m_xy) - self.d_xy
            err_xy = np.clip(err_xy, 0, None).max()

            err_z = np.abs(A_z(self.P) - self.m_z) - self.d_z
            err_z = np.clip(err_z, 0, None).max()

            err_I = np.abs(A_I(self.P) - self.dirac).max()

            w, v = np.linalg.eigh(self.P)
            err_rank1 = np.linalg.norm(w[:-1])

            n = len(self.P) // 2
            obj = self.P[n - 1, n - 1]
            obj += self.eps * self.P[n - 1, 2 * n - 1]
            obj += self.eps * self.P[2 * n - 1, n - 1]
            obj = np.real(obj)
            self.pbar.set_postfix(err_I='{0:.2E}'.format(err_I),
                                  err_xy='{0:.2E}'.format(err_xy),
                                  err_z='{0:.2E}'.format(err_z),
                                  err_rank1='{0:.2E}'.format(err_rank1),
                                  obj='{0:.2E}'.format(obj))

    def _output(self):
        w, v = np.linalg.eigh(self.P)
        x = v[:, -1] * w[-1]**0.5

        n = self.P.shape[0] // 2
        a = x[:n] * np.sign(x[0])
        b = x[n:] * np.sign(x[0])

        return a, b
