import numpy as np
import sigpy as sp
import linop
import prox
import sigpy.mri.rf as rf


def dzrf(n=64, tb=4, ptype='ex', d1=0.01, d2=0.01,
         m=1000, max_iter=None, sigma=None, eps=None, return_P=False):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    bands = [[-np.pi, -(1 + w) * tb / n * np.pi],
             [-(1 - w) * tb / n * np.pi, (1 - w) * tb / n * np.pi],
             [(1 + w) * tb / n * np.pi, np.pi]]

    if ptype == 'ex':
        m_xy_vals = [0, lambda omega: np.exp(1j * omega * (n - 1) / 2), 0]
        m_xy_errs = [d2, d1, d2]
        d1_z = (1 - (1 - d1)**2)**0.5
        d2_z = 1 - (1 - d2**2)**0.5
        m_z_vals = [1 - d2_z / 2, 0, 1 - d2_z / 2]
        m_z_errs = [d2_z / 2, d1_z, d2_z / 2]
        if sigma is None:
            sigma = 1000
        if eps is None:
            eps = 0
        if max_iter is None:
            max_iter = 2000
    elif ptype == 'ex-minphase':
        m_xy_vals = [0, 0, 0]
        m_xy_errs = [d2, 1, d2]
        d1_z = (1 - (1 - d1)**2)**0.5
        d2_z = 1 - (1 - d2**2)**0.5
        m_z_vals = [1 - d2_z / 2, 0, 1 - d2_z / 2]
        m_z_errs = [d2_z / 2, d1_z, d2_z / 2]
        if sigma is None:
            sigma = 100
        if eps is None:
            eps = 1
        if max_iter is None:
            max_iter = 12000
    elif ptype == 'inv':
        m_xy_vals = [0, 0, 0]
        d1_xy = (1 - (1 - d1)**2)**0.5
        d2_xy = (1 - (1 - d2)**2)**0.5
        m_xy_errs = [d2_xy, d1_xy, d2_xy]
        m_z_vals = [1 - d2 / 2, -1 + d1 / 2, 1 - d2 / 2]
        m_z_errs = [d2 / 2, d1 / 2, d2 / 2]
        if sigma is None:
            sigma = 1000
        if eps is None:
            eps = 1
        if max_iter is None:
            max_iter = 12000

    m_xy, d_xy = bands_to_arrays(m, bands, m_xy_vals, m_xy_errs)
    m_z, d_z = bands_to_arrays(m, bands, m_z_vals, m_z_errs)

    app = DesignPaulynomials(
        n, m_xy, m_z, d_xy, d_z,
        return_P=return_P, max_iter=max_iter, sigma=sigma, eps=eps)

    if return_P:
        return app.run()
    else:
        a, b = app.run()
        b1 = rf.ab2rf(a, b)
        return b1


def dzmbrf(n=256, tb=4, ptype='ex', d1=0.01, d2=0.01,
           n_bands=3, band_sep=12, phs_0_pt='None',
           m=1000, max_iter=None, sigma=None, eps=None):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    if phs_0_pt != 'None':
        phs = rf.multiband.mb_phs_tab(n_bands, phs_0_pt)
    else:
        phs = np.zeros(n_bands)

    def get_m_xy_func(i):
        return lambda omega: np.exp(
            -1j * phs[-i - 1] + 1j * omega * (n - 1) / 2)

    bands = []
    m_xy_vals = []
    m_xy_errs = []
    m_z_vals = []
    m_z_errs = []
    # First stop band
    center = 2 * np.pi * band_sep * (- (n_bands - 1) / 2) / n
    bands.append([-np.pi, center - (1 + w) * tb / n * np.pi])
    m_xy_vals.append(0)
    m_xy_errs.append(d2)
    m_z_vals.append(1)
    m_z_errs.append(1 - (1 - d2**2)**0.5)

    for i in range(n_bands):
        center = 2 * np.pi * band_sep * (i - (n_bands - 1) / 2) / n
        # Pass band
        bands.append([center - (1 - w) * tb / n * np.pi,
                      center + (1 - w) * tb / n * np.pi])
        m_xy_vals.append(get_m_xy_func(i))
        m_xy_errs.append(d1)
        m_z_vals.append(0)
        m_z_errs.append((1 - (1 - d1)**2)**0.5)

        # Stop band
        if i == n_bands - 1:
            end = np.pi
        else:
            next_center = 2 * np.pi * band_sep * (
                i + 1 - (n_bands - 1) / 2) / n
            end = next_center - (1 + w) * tb / n * np.pi

        bands.append([center + (1 + w) * tb / n * np.pi, end])
        m_xy_vals.append(0)
        m_xy_errs.append(d2)
        m_z_vals.append(1)
        m_z_errs.append(1 - (1 - d2**2)**0.5)

    if sigma is None:
        sigma = 1000
    if eps is None:
        eps = 0
    if max_iter is None:
        max_iter = 2000

    m_xy, d_xy = bands_to_arrays(m, bands, m_xy_vals, m_xy_errs)
    m_z, d_z = bands_to_arrays(m, bands, m_z_vals, m_z_errs)

    a, b = DesignPaulynomials(n, m_xy, m_z, d_xy, d_z,
                              max_iter=max_iter, sigma=sigma, eps=eps).run()
    b1 = rf.ab2rf(a, b)
    return b1


def bands_to_arrays(m, bands, vals, errs):
    """Convert M_xy and M_z band specifications to arrays.

    bands (list of bands): list of frequency bands, specified by
         starting and end points in radians between -pi to pi.
    vals (list of floats): desired magnitude response for each band.
        Must have the same length as bands. For example,
        [0, 1, 0].
    linphases (list of floats): desired linear phase for each band.
        Must have the same length as bands.
    errs (list of floats): maximum deviation from specified profile.
    """
    omega = 2 * np.pi * (np.arange(m) - m // 2) / m
    v = np.zeros(m, dtype=np.complex)
    d = np.ones(m)

    for band, val, delta, in zip(bands, vals, errs):
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


def P_to_ab(P, return_err=False):
    w, v = np.linalg.eigh(P)
    x = v[:, -1] * w[-1]**0.5
    err = np.linalg.norm(w[:-1])

    n = P.shape[0] // 2
    a = x[:n] * np.sign(x[0])
    b = x[n:] * np.sign(x[0])
    if return_err:
        return a, b, err
    else:
        return a, b


def ab_to_P(a, b):
    x = np.concatenate([a, b])
    return np.outer(x, x.conjugate())


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
        errs = [0.01, 0.01, 0.01]

    Returns:
        array, array: alpha and beta polynomials of length n.

    """
    def __init__(self, n, m_xy, m_z, d_xy, d_z,
                 sigma=100, eps=0, max_iter=10000, return_P=False):

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
        As = [A_xy, A_xyc, A_z, A_I]
        A = sp.linop.Vstack(As)

        # Create proximal operators
        dirac = np.zeros(2 * n - 1)
        dirac[n - 1] = 1

        proxf_xy = sp.prox.LInfProj(A_xy.oshape, d_xy, m_xy)
        proxf_xyc = sp.prox.LInfProj(A_xyc.oshape, d_xy, np.conj(m_xy))
        proxf_z = sp.prox.LInfProj(A_z.oshape, d_z, m_z)
        proxf_I = sp.prox.LInfProj(A_I.oshape, 0, dirac)
        proxf = sp.prox.Stack([
            proxf_xy, proxf_xyc, proxf_z, proxf_I])
        proxfc = sp.prox.Conj(proxf)

        proxg = prox.Objective((2 * n, 2 * n), eps)

        # Get step-size
        sigma_xy = sigma / (4 * n * m)
        sigma_z = sigma / (2 * n * m)
        sigma_I = sigma / (2 * n)
        tau = 1 / sigma
        sigma = np.concatenate(
            [np.full(A_xy.oshape, sigma_xy),
             np.full(A_xyc.oshape, sigma_xy),
             np.full(A_z.oshape, sigma_z),
             np.full(A_I.oshape, sigma_I)])

        # Get variables
        self.P = np.zeros((2 * n, 2 * n), dtype=np.complex)
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.P, u, tau, sigma,
            max_iter=max_iter)
        self.m_xy = m_xy
        self.m_z = m_z
        self.d_xy = d_xy
        self.d_z = d_z
        self.As = As
        self.eps = eps
        self.dirac = dirac
        self.return_P = return_P
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            A_xy, A_xyc, A_z, A_I = self.As
            a, b, err_rank1 = P_to_ab(self.P, return_err=True)
            err_xy = np.max(
                np.clip(np.abs(A_xy(self.P) - self.m_xy) - self.d_xy, 0, None))
            err_z = np.max(
                np.clip(np.abs(A_z(self.P) - self.m_z) - self.d_z, 0, None))
            err_I = np.max(np.abs(A_I(self.P) - self.dirac))

            n = len(self.P) // 2
            obj = self.P[n - 1, n - 1]
            obj += self.eps * self.P[n - 1, 2 * n - 1]
            obj += self.eps * self.P[2 * n - 1, n - 1]
            obj = np.real(obj)
            self.pbar.set_postfix(err_rank1='{0:.2E}'.format(err_rank1),
                                  err_I='{0:.2E}'.format(err_I),
                                  err_xy='{0:.2E}'.format(err_xy),
                                  err_z='{0:.2E}'.format(err_z),
                                  obj='{0:.2E}'.format(obj))

    def _output(self):
        if self.return_P:
            return self.P
        else:
            return P_to_ab(self.P)


class DesignMultibandPaulynomials(sp.app.App):
    """Design Shinnar-Le-Roux polynomials given an Mxy profile.

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
        errs = [0.01, 0.01, 0.01]

    Returns:
        array, array: alpha and beta polynomials of length n.

    """
    def __init__(self, n, tb, d1=0.01, d2=0.01, n_bands=3, band_sep=12,
                 m=2000, sigma=1000, eps=1, max_iter=10000, return_P=False):
        # Create bands
        dinf = rf.dinf(d1, d2)
        w = dinf / tb
        omega = 2 * np.pi * (np.arange(m) - m // 2) / m
        d_xy = np.ones(m)
        m_z = np.zeros(m)
        d_z = np.ones(m)
        # First stop band
        center = 2 * np.pi * band_sep * (- (n_bands - 1) / 2) / n
        idx = (omega >= -np.pi) & (omega <= center - (1 + w) * tb / n * np.pi)
        d_xy[idx] = d2
        m_z[idx] = 1
        d_z[idx] = (1 - (1 - d2**2)**0.5)

        linphase = np.zeros([m, n_bands], dtype=np.complex)
        for i in range(n_bands):
            center = 2 * np.pi * band_sep * (i - (n_bands - 1) / 2) / n
            # Pass band
            idx = (omega >= center - (1 - w) * tb / n * np.pi) & \
                (omega <= center + (1 - w) * tb / n * np.pi)
            linphase[idx, i] = np.exp(1j * omega[idx] * (n - 1) / 2)
            d_xy[idx] = d1
            m_z[idx] = 0
            d_z[idx] = (1 - (1 - d1)**2)**0.5

            # Stop band
            if i == n_bands - 1:
                end = np.pi
            else:
                next_center = 2 * np.pi * band_sep * (
                    i + 1 - (n_bands - 1) / 2) / n
                end = next_center - (1 + w) * tb / n * np.pi

            idx = (omega >= center + (1 + w) * tb / n * np.pi) & \
                (omega <= end)
            d_xy[idx] = d2
            m_z[idx] = 1
            d_z[idx] = (1 - (1 - d2**2)**0.5)

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
        As = [A_xy, A_xyc, A_z, A_I]
        A = sp.linop.Vstack(As)

        # Create proximal operators
        dirac = np.zeros(2 * n - 1)
        dirac[n - 1] = 1

        proxf_xy = sp.prox.LInfProj(A_xy.oshape, d_xy)
        proxf_xyc = sp.prox.LInfProj(A_xyc.oshape, d_xy)
        proxf_z = sp.prox.LInfProj(A_z.oshape, d_z, m_z)
        proxf_I = sp.prox.LInfProj(A_I.oshape, 0, dirac)
        proxf = sp.prox.Stack([
            proxf_xy, proxf_xyc, proxf_z, proxf_I])
        proxfc = sp.prox.Conj(proxf)

        proxg = sp.prox.Stack(
            [sp.prox.UnitaryTransform(prox.Objective((2 * n, 2 * n), eps),
                                      sp.linop.Reshape((2 * n, 2 * n), [4 * n**2])),
             sp.prox.NoOp([2 * n_bands])])

        # Get step-size
        sigma_xy = sigma / (4 * n * m)
        sigma_z = sigma / (2 * n * m)
        sigma_I = sigma / (2 * n)
        tau = 1 / sigma
        sigma = np.concatenate(
            [np.full(A_xy.oshape, sigma_xy),
             np.full(A_xyc.oshape, sigma_xy),
             np.full(A_z.oshape, sigma_z),
             np.full(A_I.oshape, sigma_I)])

        # Get variables
        self.x = np.zeros([4 * n**2 + 2 * n_bands], dtype=np.complex)
        self.P = self.x[:4 * n**2].reshape((2 * n, 2 * n))
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.x, u, tau, sigma,
            max_iter=max_iter)
        self.m_z = m_z
        self.d_xy = d_xy
        self.d_z = d_z
        self.As = As
        self.eps = eps
        self.dirac = dirac
        self.return_P = return_P
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            A_xy, A_xyc, A_z, A_I = self.As
            a, b, err_rank1 = P_to_ab(self.P, return_err=True)
            err_xy = np.max(
                np.clip(np.abs(A_xy(self.x)) - self.d_xy, 0, None))
            err_z = np.max(
                np.clip(np.abs(A_z(self.x) - self.m_z) - self.d_z, 0, None))
            err_I = np.max(np.abs(A_I(self.x) - self.dirac))

            n = len(self.P) // 2
            obj = self.P[n - 1, n - 1]
            obj += self.eps * self.P[n - 1, 2 * n - 1]
            obj += self.eps * self.P[2 * n - 1, n - 1]
            obj = np.real(obj)
            self.pbar.set_postfix(err_rank1='{0:.2E}'.format(err_rank1),
                                  err_I='{0:.2E}'.format(err_I),
                                  err_xy='{0:.2E}'.format(err_xy),
                                  err_z='{0:.2E}'.format(err_z),
                                  obj='{0:.2E}'.format(obj))

    def _output(self):
        if self.return_P:
            return self.P
        else:
            return P_to_ab(self.P)
