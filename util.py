import numpy as np
import matplotlib.pyplot as plt
import sigpy.mri.rf as rf


def plot_rf(b1, balanced=False, magphase=False, m=1000,
            omega_range=[-np.pi, np.pi]):
    figs = []
    n = len(b1)
    fig, ax = plt.subplots()
    ax.plot(b1.real, label=r'$B_{1, \mathrm{x}}$')
    ax.plot(b1.imag, label=r'$B_{1, \mathrm{y}}$')
    ax.set_title(r'$B_1$ (Energy={:.3g})'.format(np.sum(np.abs(b1)**2)))
    ax.set_xlabel('Time')
    ax.legend()
    figs.append(fig)

    omega = np.linspace(omega_range[0], omega_range[1], m)
    x = omega * n / (2 * np.pi)
    a, b = rf.abrm(b1, x, balanced=balanced)
    m_xy = 2 * b.conjugate() * a
    m_z = (a * a.conjugate() - b * b.conjugate()).real

    if magphase:
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
    else:
        fig, ax = plt.subplots()
        ax.set_title(r'$M_{\mathrm{xy}}$')
        ax.set_xlabel(r'$\omega$ [radian]')
        ax.plot(omega, m_xy.real, label=r'$M_{\mathrm{x}}$')
        ax.plot(omega, m_xy.imag, label=r'$M_{\mathrm{y}}$')
        ax.legend()
        figs.append(fig)

    fig, ax = plt.subplots()
    ax.set_title(r'$M_{\mathrm{z}}$')
    ax.set_xlabel(r'$\omega$ [radian]')
    ax.plot(omega, m_z)
    figs.append(fig)

    return figs
