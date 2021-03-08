import numpy as np
import matplotlib.pyplot as plt
from . import transform


def plot_pulse(pulse, m=1000, ptype='ex', phase='linear',
               omega_range=[-np.pi, np.pi],
               linewidth=2, fontsize='x-large', labelsize='large'):
    figs = []
    n = len(pulse)
    fig, ax = plt.subplots()
    ax.plot(pulse.real, label=r'$B_{1, \mathrm{x}}$', linewidth=linewidth)
    ax.plot(pulse.imag, label=r'$B_{1, \mathrm{y}}$', linewidth=linewidth)
    ax.set_title(r'$B_1$ (Energy={0:.3g}, Peak={1:.3g})'.format(
        np.sum(np.abs(pulse)**2), np.max(np.abs(pulse))), fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    ax.xaxis.set_tick_params(labelsize=labelsize)
    figs.append(fig)

    omega = np.linspace(omega_range[0], omega_range[1], m)
    psi_z = np.exp(-1j * np.outer(omega, np.arange(n)))
    a, b = transform.forward_slr(pulse)
    alpha = psi_z @ a
    beta = psi_z @ b

    if ptype == 'se':
        m_xy = beta**2
        m_xy *= np.exp(1j * omega * (n - 1))
        fig, ax = plt.subplots()
        ax.set_title(r'$M_{\mathrm{xy}}$')
        ax.set_xlabel(r'$\omega$ [radian]')
        ax.plot(omega, np.real(m_xy), label=r'$M_{\mathrm{x}}$', linewidth=linewidth)
        ax.plot(omega, np.imag(m_xy), label=r'$M_{\mathrm{y}}$', linewidth=linewidth)
        ax.legend(fontsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.xaxis.set_tick_params(labelsize=labelsize)
        figs.append(fig)
    else:
        m_xy = 2 * alpha.conjugate() * beta
        m_z = np.abs(alpha)**2 - np.abs(beta)**2
        if phase == 'linear':
            m_xy *= np.exp(1j * omega * n / 2)

        fig, ax = plt.subplots()
        ax.set_title(r'$|M_{\mathrm{xy}}|$', fontsize=fontsize)
        ax.set_xlabel(r'$\omega$ [radian]', fontsize=fontsize)
        ax.plot(omega, np.abs(m_xy), linewidth=linewidth)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.xaxis.set_tick_params(labelsize=labelsize)
        figs.append(fig)

        fig, ax = plt.subplots()
        ax.set_title(r'$\angle M_{\mathrm{xy}}$', fontsize=fontsize)
        ax.set_xlabel(r'$\omega$ [radian]', fontsize=fontsize)
        ax.plot(omega, np.angle(m_xy), linewidth=linewidth)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.xaxis.set_tick_params(labelsize=labelsize)
        figs.append(fig)

        fig, ax = plt.subplots()
        ax.set_title(r'$M_{\mathrm{z}}$', fontsize=fontsize)
        ax.set_xlabel(r'$\omega$ [radian]', fontsize=fontsize)
        ax.plot(omega, m_z, linewidth=linewidth)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.xaxis.set_tick_params(labelsize=labelsize)
        figs.append(fig)

    return figs


def plot_slr_pulses(pulse_slr, pulse_slfrank,
                    m=1000, ptype='ex', phase='linear',
                    omega_range=[-np.pi, np.pi],
                    fontsize='x-large', labelsize='large'):
    n = len(pulse_slr)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(pulse_slr.real,
                   linewidth=0.5,
                   label='SLR',
                   color='tab:orange')
    axs[0][0].plot(pulse_slfrank.real,
                   linewidth=0.5,
                   label='SLfRank',
                   color='tab:blue')
    axs[0][0].set_title(r'$B_{1}$')
    axs[0][0].set_xlabel('Time')
    axs[0][0].legend()

    omega = np.linspace(omega_range[0], omega_range[1], m)
    psi_z = np.exp(-1j * np.outer(omega, np.arange(n)))

    a_slr, b_slr = transform.forward_slr(pulse_slr)
    alpha_slr = psi_z @ a_slr
    beta_slr = psi_z @ b_slr

    a_slfrank, b_slfrank = transform.forward_slr(pulse_slfrank)
    alpha_slfrank = psi_z @ a_slfrank
    beta_slfrank = psi_z @ b_slfrank

    if ptype == 'se':
        m_xy_slr = beta_slr**2
        m_xy_slr *= np.exp(1j * omega * (n - 1))
        m_z_slr = 2 * np.imag(alpha_slr * beta_slr)
        m_xy_slfrank = beta_slfrank**2
        m_xy_slfrank *= np.exp(1j * omega * (n - 1))
        m_z_slfrank = 2 * np.imag(alpha_slfrank * beta_slfrank)

        axs[1][0].set_title(r'$M_{\mathrm{x}}$')
        axs[1][0].set_xlabel(r'$\omega$ [radian]')
        axs[1][0].plot(omega, np.real(m_xy_slr),
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[1][0].plot(omega, np.real(m_xy_slfrank),
                       linewidth=0.5,
                       label='SLfRank',
                       color='tab:blue')

        axs[1][1].set_title(r'$M_{\mathrm{y}}$')
        axs[1][1].set_xlabel(r'$\omega$ [radian]')
        axs[1][1].plot(omega, np.imag(m_xy_slr),
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[1][1].plot(omega, np.imag(m_xy_slfrank),
                       linewidth=0.5,
                       label='SLfRank',
                       color='tab:blue')

        axs[0][1].set_title(r'$M_{\mathrm{z}}$')
        axs[0][1].set_xlabel(r'$\omega$ [radian]')
        axs[0][1].plot(omega, m_z_slr,
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[0][1].plot(omega, m_z_slfrank,
                       linewidth=0.5,
                       label='SLfRank',
                       color='tab:blue')
    else:
        m_xy_slr = 2 * alpha_slr.conjugate() * beta_slr
        m_z_slr = np.abs(alpha_slr)**2 - np.abs(beta_slr)**2
        m_xy_slfrank = 2 * alpha_slfrank.conjugate() * beta_slfrank
        m_z_slfrank = np.abs(alpha_slfrank)**2 - np.abs(beta_slfrank)**2
        if phase == 'linear':
            m_xy_slr *= np.exp(1j * omega * n / 2)
            m_xy_slfrank *= np.exp(1j * omega * n / 2)

        axs[1][0].set_title(r'$|M_{\mathrm{xy}}|$')
        axs[1][0].set_xlabel(r'$\omega$ [radian]')
        axs[1][0].plot(omega, np.abs(m_xy_slr),
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[1][0].plot(omega, np.abs(m_xy_slfrank),
                       linewidth=0.5,
                       label=r'SLfRank',
                       color='tab:blue')

        axs[1][1].set_title(r'$\angle M_{\mathrm{xy}}$')
        axs[1][1].set_xlabel(r'$\omega$ [radian]')
        axs[1][1].plot(omega, np.angle(m_xy_slr),
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[1][1].plot(omega, np.angle(m_xy_slfrank),
                       linewidth=0.5,
                       label=r'SLfRank',
                       color='tab:blue')

        axs[0][1].set_title(r'$M_{\mathrm{z}}$')
        axs[0][1].set_xlabel(r'$\omega$ [radian]')
        axs[0][1].plot(omega, m_z_slr,
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[0][1].plot(omega, m_z_slfrank,
                       linewidth=0.5,
                       label=r'SLfRank',
                       color='tab:blue')

    return fig
