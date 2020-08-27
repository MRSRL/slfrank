import numpy as np
import matplotlib.pyplot as plt
import sigpy.mri.rf as rf


def plot_rf(b1, m=1000,
            refocus_linphase=0,
            omega_range=[-np.pi, np.pi]):
    figs = []
    n = len(b1)
    fig, ax = plt.subplots()
    ax.plot(b1.real, label=r'$B_{1, \mathrm{x}}$')
    ax.plot(b1.imag, label=r'$B_{1, \mathrm{y}}$')
    ax.set_title(r'$B_1$ (Energy={0:.3g}, Max={1:.3g})'.format(
        np.sum(np.abs(b1)**2), np.max(np.abs(b1))))
    ax.set_xlabel('Time')
    ax.legend()
    figs.append(fig)

    omega = np.linspace(omega_range[0], omega_range[1], m)
    x = omega * n / (2 * np.pi)
    g = np.ones(n) * 2 * np.pi / n
    a, b = rf.abrm_hp(b1, g, x)
    m_xy = 2 * a.conjugate() * b
    m_z = (a * a.conjugate() - b * b.conjugate()).real
    m_xy *= np.exp(1j * omega * refocus_linphase)

    fig, ax = plt.subplots()
    ax.set_title(r'$|M_{\mathrm{xy}}|$')
    ax.set_xlabel(r'$\omega$ [radian]')
    ax.plot(omega, np.abs(m_xy))
    figs.append(fig)

    fig, ax = plt.subplots()
    ax.set_title(r'$\angle M_{\mathrm{xy}}$')
    ax.set_xlabel(r'$\omega$ [radian]')
    ax.plot(omega, np.angle(m_xy))
    figs.append(fig)

    fig, ax = plt.subplots()
    ax.set_title(r'$M_{\mathrm{z}}$')
    ax.set_xlabel(r'$\omega$ [radian]')
    ax.plot(omega, m_z)
    figs.append(fig)

    return figs
