import numpy as np
import sigpy as sp
import linop
import sigpy.mri.rf as rf


def dzrf(n=64, tb=4, ptype='ex', d1=0.01, d2=0.01,
         m=1000, max_iter=10000):
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    bands = [[-np.pi, -(1 + w) * tb / n * np.pi],
             [-(1 - w) * tb / n * np.pi, (1 - w) * tb / n * np.pi],
             [(1 + w) * tb / n * np.pi, np.pi]]

    if ptype == 'ex':
        m_xy_mags = [0, 1, 0]
        m_xy_linphases = [0, (n - 1) / 2, 0]
        delta_xys = [d2, d1, d2]
        m_z_vals = [1, 0, 1]
        delta_zs = [1 - (1 - d2**2)**0.5, (1 - (1 - d1)**2)**0.5, 1 - (1 - d2**2)**0.5]
        sigma = 1
        eps = 0
    elif ptype == 'ex-minphase':
        m_xy_mags = [0, 0, 0]
        m_xy_linphases = [0, 0, 0]
        delta_xys = [d2, 1, d2]
        m_z_vals = [1, 0, 1]
        delta_zs = [1 - (1 - d2**2)**0.5, (1 - (1 - d1)**2)**0.5, 1 - (1 - d2**2)**0.5]
        sigma = 100
        eps = 1
    elif ptype == 'inv':
        m_xy_mags = [0, 0, 0]
        m_xy_linphases = [0, 0, 0]
        delta_xys = [(1 - (1 - d2)**2)**0.5, (1 - (1 - d1)**2)**0.5, (1 - (1 - d2)**2)**0.5]
        m_z_vals = [1, -1, 1]
        delta_zs = [d2, d1, d2]
        sigma = 1000
        eps = 1

    m_xy, m_z, d_xy, d_z = bands_to_arrays(
        m, bands, m_xy_mags, m_xy_linphases, delta_xys, m_z_vals, delta_zs)
    a, b = DesignPaulynomials(n, m_xy, m_z, d_xy, d_z, max_iter=max_iter, sigma=sigma, eps=eps).run()
    b1 = rf.ab2rf(a, b)
    return b1


def bands_to_arrays(m, bands, m_xy_mags, m_xy_linphases, delta_xys,
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

    for band, m_xy_mag, m_xy_linphase, delta_xy, m_z_val, delta_z, in zip(
            bands, m_xy_mags, m_xy_linphases, delta_xys, m_z_vals, delta_zs):
        i = (omega >= band[0]) & (omega <= band[1])
        m_xy[i] = m_xy_mag * np.exp(1j * omega[i] * m_xy_linphase)
        d_xy[i] = delta_xy

        m_z[i] = m_z_val
        d_z[i] = delta_z

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
                 sigma=1000, eps=1, max_iter=10000, m=1000):

        # Create linear operators
        S_aa = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n)))
        S_ba = sp.linop.Slice((2 * n, 2 * n), (slice(n, 2 * n), slice(n)))
        S_bb = sp.linop.Slice(
            (2 * n, 2 * n), (slice(n, 2 * n), slice(n, 2 * n)))

        D = linop.DiagSum(n)
        R = sp.linop.Resize((m, ), D.oshape)
        F = sp.linop.FFT((m, ))

        A_I = m**0.5 * F * R * D * (S_aa + S_bb)
        A_xy = m**0.5 * F * R * D * (2 * S_ba)
        A_z = m**0.5 * F * R * D * (S_aa - S_bb)
        A_obj = sp.linop.Slice((2 * n, 2 * n), ([n - 1], [n - 1])) + eps * (
            sp.linop.Slice((2 * n, 2 * n), ([n - 1], [2 * n - 1])) + sp.linop.Slice((2 * n, 2 * n), ([2 * n - 1], [n - 1])))
        As = [A_I, A_xy, A_z, A_obj]
        A = sp.linop.Vstack(As)

        # Create proximal operators
        def proxfc(alpha, u):
            u_I, u_xy, u_z, u_obj = np.split(u, [m, 2 * m, 3 * m])
            alpha_I, alpha_xy, alpha_z, alpha_obj = np.split(
                alpha, [m, 2 * m, 3 * m])

            u_I[:] = u_I - alpha_I
            u_xy[:] = sp.soft_thresh(alpha_xy * d_xy, u_xy - alpha_xy * m_xy)
            u_z[:] = sp.soft_thresh(alpha_z * d_z, u_z - alpha_z * m_z)
            u_obj[:] = -1
            return u

        def proxg(alpha, X):
            w, v = np.linalg.eigh(X)
            w[w < 0] = 0
            return (v * w) @ v.conjugate().T

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
            proxfc, proxg, A, A.H, self.P, u, tau, sigma, max_iter=max_iter, tol=-1)
        self.m_xy = m_xy
        self.m_z = m_z
        self.d_xy = d_xy
        self.d_z = d_z
        self.As = As
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            A_I, A_xy, A_z, A_obj = self.As
            err_I = np.abs(A_I(self.P) - 1).max()

            err_xy = np.abs(A_xy(self.P) - self.m_xy) - self.d_xy
            err_xy = np.clip(err_xy, 0, None).max()

            err_z = np.abs(A_z(self.P) - self.m_z) - self.d_z
            err_z = np.clip(err_z, 0, None).max()

            w, v = np.linalg.eigh(self.P)
            err_rank1 = np.linalg.norm(w[:-1])

            obj = np.sum(A_obj(self.P)).real
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


class DesignSlrMxy(sp.app.App):
    """Design Shinnar-Le-Roux polynomials given an Mxy profile.

    Args:
        n (int): number of hard pulses.
        bands (list of bands): list of frequency bands, specified by
             starting and end points in radians between -pi to pi.
        vals (list of floats): desired magnitude response for each band.
            Must have the same length as bands. For example,
            [0, 1, 0].
        linphases (list of floats): desired linear phase for each band.
            Must have the same length as bands.
        deltas (list of floats): maximum deviation from specified profile.
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
    def __init__(self, n, bands, vals, linphases, deltas,
                 sigma=5000, max_iter=5000, m=1000):

        # Create ideal profile
        omega = 2 * np.pi * (np.arange(m) - m // 2) / m
        m_xy = np.zeros(m, dtype=np.complex)
        d = np.zeros(m)
        idx = np.zeros(m, dtype=bool)

        for band, val, linphase, delta, in zip(bands, vals, linphases, deltas):
            i = (omega >= band[0]) & (omega <= band[1])
            m_xy[i] = val * np.exp(-1j * omega[i] * linphase)
            d[i] = delta
            idx |= i

        m_xy = m_xy[idx]
        d = d[idx]

        # Create linear operators
        S_aa = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n)))
        S_ba = sp.linop.Slice((2 * n, 2 * n), (slice(n, 2 * n), slice(n)))
        S_bb = sp.linop.Slice(
            (2 * n, 2 * n), (slice(n, 2 * n), slice(n, 2 * n)))

        D = linop.DiagSum(n)
        R = sp.linop.Resize((m, ), D.oshape)
        F = sp.linop.FFT((m, ))
        S = sp.linop.Slice((m, ), idx)

        A1 = m**0.5 * F * R * D * (S_aa + S_bb)
        A2 = m**0.5 * S * F * R * D * (2 * S_ba)
        A3 = sp.linop.Slice((2 * n, 2 * n), (slice(1), slice(1)))

        A = sp.linop.Vstack([A1, A2, A3])

        # Create proximal operators
        def proxfc(alpha, u):
            u = u.copy()
            u1 = u[:m]
            u2 = u[m:-1]
            u3 = u[-1:]
            alpha1 = alpha[:m]
            alpha2 = alpha[m:-1]

            u1[:] = u1 - alpha1
            u2[:] = sp.soft_thresh(alpha2 * d, u2 - alpha2 * m_xy)
            u3[:] = -1
            return u

        def proxg(alpha, X):
            w, v = np.linalg.eigh(X)
            w[w < 0] = 0
            return (v * w) @ v.conjugate().T

        # Get step-size
        L1 = sp.app.MaxEig(A1.H * A1, dtype=np.complex).run()
        L2 = sp.app.MaxEig(A2.H * A2, dtype=np.complex).run()
        L3 = sp.app.MaxEig(A3.H * A3, dtype=np.complex).run()
        sigma1 = np.full(sp.prod(A1.oshape), sigma / L1)
        sigma2 = np.full(sp.prod(A2.oshape), sigma / L2)
        sigma3 = np.full(sp.prod(A3.oshape), sigma / L3)
        tau = 1 / sigma / 3
        sigma = np.concatenate([sigma1, sigma2, sigma3])

        self.P = np.zeros(A.ishape, dtype=np.complex)
        self.P[0, 0] = 1
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.P, u, tau, sigma, max_iter=max_iter)
        self.m_xy = m_xy
        self.d = d
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            d1_resid = np.abs(self.A1(self.P) - 1).max()
            dxy_resid = np.abs(self.A2(self.P) - self.m_xy) - self.d
            dxy_resid = np.clip(dxy_resid, 0, None).max()
            w, v = np.linalg.eigh(self.P)
            rank1_err = np.linalg.norm(w[:-1])
            obj = np.sum(self.A3(self.P)).real
            self.pbar.set_postfix(d1_resid=d1_resid, dxy_resid=dxy_resid,
                                  obj=obj, rank1_err=rank1_err)

    def _output(self):
        a, b = P_to_ab(self.P)
        return a, b


class DesignSlrMxyMinPhase(sp.app.App):
    """Design Shinnar-Le-Roux polynomials given an Mxy profile.

    Args:
        n (int): number of hard pulses.
        bands (list of bands): list of frequency bands, specified by
             starting and end points in radians between -pi to pi.
        vals (list of floats): desired magnitude response for each band.
            Must have the same length as bands. For example,
            [0, 1, 0].
        linphases (list of floats): desired linear phase for each band.
            Must have the same length as bands.
        deltas (list of floats): maximum deviation from specified profile.
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
    def __init__(self, n, bands, vals, deltas,
                 sigma=1, max_iter=10000, m=1000):

        # Create ideal profile
        omega = 2 * np.pi * (np.arange(m) - m // 2) / m
        m_xy = np.zeros(m)
        d = np.zeros(m)
        idx = np.zeros(m, dtype=bool)

        for band, val, delta, in zip(bands, vals, deltas):
            i = (omega >= band[0]) & (omega <= band[1])
            m_xy[i] = val
            d[i] = delta
            idx |= i

        m_xy = m_xy[idx]
        d = d[idx]

        # Create linear operators
        S_aa = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n)))
        S_ba = sp.linop.Slice((2 * n, 2 * n), (slice(n, 2 * n), slice(n)))
        S_bb = sp.linop.Slice(
            (2 * n, 2 * n), (slice(n, 2 * n), slice(n, 2 * n)))

        D = linop.DiagSum(n)
        R = sp.linop.Resize((m, ), D.oshape)
        F = sp.linop.FFT((m, ))
        S = sp.linop.Slice((m, ), idx)

        A1 = m**0.5 * F * R * D * (S_aa + S_bb)
        A2 = m**0.5 * S * F * R * D * (2 * S_ba)
        A3 = sp.linop.Vstack([
            0 * sp.linop.Slice((2 * n, 2 * n), (slice(1), slice(1))),
            sp.linop.Slice((2 * n, 2 * n), ([0, n], [n, 0]))
            ])

        A = sp.linop.Vstack([A1, A2, A3])

        # Create proximal operators
        def proxfc(alpha, u):
            u1 = u[:m]
            u2 = u[m:-3]
            u3 = u[-3:]
            alpha1 = alpha[:m]
            alpha2 = alpha[m:-3]

            u1[:] = u1 - alpha1
            u2[:] = sp.soft_thresh(alpha2 * (m_xy + d), u2)
            u3[:] = -1
            return u

        def proxg(alpha, X):
            w, v = np.linalg.eigh(X)
            w[w < 0] = 0
            return (v * w) @ v.conjugate().T

        # Get step-size
        L1 = sp.app.MaxEig(A1.H * A1, dtype=np.complex).run()
        L2 = sp.app.MaxEig(A2.H * A2, dtype=np.complex).run()
        L3 = sp.app.MaxEig(A3.H * A3, dtype=np.complex).run()
        sigma1 = np.full(sp.prod(A1.oshape), sigma / L1)
        sigma2 = np.full(sp.prod(A2.oshape), sigma / L2)
        sigma3 = np.full(sp.prod(A3.oshape), sigma / L3)
        tau = 1 / sigma / 3
        sigma = np.concatenate([sigma1, sigma2, sigma3])

        self.P = np.zeros(A.ishape, dtype=np.complex)
        self.P[0, 0] = 1
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.P, u, tau, sigma, max_iter=max_iter)
        self.m_xy = m_xy
        self.d = d
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            d1_resid = np.abs(self.A1(self.P) - 1).max()
            dxy_resid = np.abs(self.A2(self.P)) - (self.m_xy + self.d)
            dxy_resid = np.clip(dxy_resid, 0, None).max()
            w, v = np.linalg.eigh(self.P)
            rank1_err = np.linalg.norm(w[:-1])
            obj = np.sum(self.A3(self.P)).real
            self.pbar.set_postfix(d1_resid=d1_resid, dxy_resid=dxy_resid,
                                  obj=obj, rank1_err=rank1_err)

    def _output(self):
        w, v = np.linalg.eigh(self.P)
        x = v[:, -1] * w[-1]**0.5

        n = self.P.shape[0] // 2
        a = x[:n] * np.sign(x[0])
        b = x[n:] * np.sign(x[0])

        return a, b


class DesignSlrMz(sp.app.App):

    def __init__(self, n, bands, vals, deltas,
                 sigma=300, max_iter=10000, m=1000):

        # Create ideal profile
        omega = 2 * np.pi * (np.arange(m) - m // 2) / m
        m_z = np.zeros(m, dtype=np.complex)
        d = np.zeros(m)
        idx = np.zeros(m, dtype=bool)

        for band, val, delta in zip(bands, vals, deltas):
            i = (omega >= band[0]) & (omega <= band[1])
            m_z[i] = val
            d[i] = delta
            idx |= i

        m_z = m_z[idx]
        d = d[idx]

        # Create linear operators
        S_aa = sp.linop.Slice((2 * n, 2 * n), (slice(n), slice(n)))
        S_bb = sp.linop.Slice(
            (2 * n, 2 * n), (slice(n, 2 * n), slice(n, 2 * n)))

        D = linop.DiagSum(n)
        R = sp.linop.Resize((m, ), D.oshape)
        F = sp.linop.FFT((m, ))
        S = sp.linop.Slice((m, ), idx)

        A1 = m**0.5 * F * R * D * (S_aa + S_bb)
        A2 = m**0.5 * S * F * R * D * (S_aa - S_bb)
        A3 = sp.linop.Vstack([
            sp.linop.Slice((2 * n, 2 * n), (slice(1), slice(1))),
            sp.linop.Slice((2 * n, 2 * n), ([0, n], [n, 0]))
            ])

        A = sp.linop.Vstack([A1, A2, A3])

        # Create proximal operators
        def proxfc(alpha, u):
            u1 = u[:m]
            u2 = u[m:-3]
            u3 = u[-3:]
            alpha1 = alpha[:m]
            alpha2 = alpha[m:-3]

            u1[:] = u1 - alpha1
            u2[:] = sp.soft_thresh(alpha2 * d, u2 - alpha2 * m_z)
            u3[:] = -1
            return u

        def proxg(alpha, X):
            w, v = np.linalg.eigh(X)
            w[w < 0] = 0
            return (v * w) @ v.conjugate().T

        # Get step-size
        L1 = sp.app.MaxEig(A1.H * A1, dtype=np.complex).run()
        L2 = sp.app.MaxEig(A2.H * A2, dtype=np.complex).run()
        L3 = sp.app.MaxEig(A3.H * A3, dtype=np.complex).run()
        sigma1 = np.full(sp.prod(A1.oshape), sigma / L1)
        sigma2 = np.full(sp.prod(A2.oshape), sigma / L2)
        sigma3 = np.full(sp.prod(A3.oshape), sigma / L3)
        tau = 1 / sigma / 3
        sigma = np.concatenate([sigma1, sigma2, sigma3])

        self.P = np.zeros(A.ishape, dtype=np.complex)
        self.P[0, 0] = 1
        u = np.zeros(A.oshape, dtype=np.complex)
        alg = sp.alg.PrimalDualHybridGradient(
            proxfc, proxg, A, A.H, self.P, u, tau, sigma, max_iter=max_iter)
        self.m_z = m_z
        self.d = d
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3

        super().__init__(alg)

    def _summarize(self):
        if self.show_pbar:
            d1_resid = np.abs(self.A1(self.P) - 1).max()
            dz_resid = np.abs(self.A2(self.P) - self.m_z) - self.d
            dz_resid = np.clip(dz_resid, 0, None).max()
            obj = np.sum(self.A3(self.P)).real
            w, v = np.linalg.eigh(self.P)
            rank1_err = np.linalg.norm(w[:-1])
            self.pbar.set_postfix(d1_resid=d1_resid, dz_resid=dz_resid,
                                  obj=obj, rank1_err=rank1_err)

    def _output(self):
        w, v = np.linalg.eigh(self.P)
        x = v[:, -1] * w[-1]**0.5

        n = self.P.shape[0] // 2
        a = x[:n] * np.sign(x[0])
        b = x[n:] * np.sign(x[0])

        return a, b
