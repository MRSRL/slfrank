import numpy as np
import matplotlib.pyplot as plt
import sigpy.mri.rf as rf
from . import transform
from . import design


def plot_slr_pulses(pulse_slr, pulse_slfrank,
                    m=1000, ptype='ex', phase='linear',
                    omega_range=[-np.pi, np.pi],
                    tb=4, d1=0.01, d2=0.01,
                    fontsize='x-large', labelsize='large'):
    n = len(pulse_slr)
    dinf = rf.dinf(d1, d2)
    w = dinf / tb
    bands = [[max(-np.pi, omega_range[0]), -(1 + w) * tb / n * np.pi],
             [-(1 - w) * tb / n * np.pi, (1 - w) * tb / n * np.pi],
             [(1 + w) * tb / n * np.pi, min(np.pi, omega_range[1])]]
    boundaries = [-(1 + w) * tb / n * np.pi,
             -(1 - w) * tb / n * np.pi,
             (1 - w) * tb / n * np.pi,
             (1 + w) * tb / n * np.pi,
             ]

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
        axs[1][0].vlines(x=boundaries,
                         ymin=min(np.amin(np.real(m_xy_slr)), np.amin(np.real(m_xy_slfrank))),
                         ymax=max(np.amax(np.real(m_xy_slr)), np.amax(np.real(m_xy_slfrank))),
                         colors='gray',
                         linestyle='dotted',
                         linewidth=0.5,
                         label='Band Boundaries')
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
        axs[1][1].vlines(x=boundaries,
                         ymin=min(np.amin(np.imag(m_xy_slr)), np.amin(np.imag(m_xy_slfrank))),
                         ymax=max(np.amax(np.imag(m_xy_slr)), np.amax(np.imag(m_xy_slfrank))),
                         colors='gray',
                         linestyle='dotted',
                         linewidth=0.5,
                         label='Band Boundaries')
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
        axs[0][1].vlines(x=boundaries,
                         ymin=min(np.amin(m_z_slr), np.amin(m_z_slfrank)),
                         ymax=max(np.amax(m_z_slr), np.amax(m_z_slfrank)),
                         colors='gray',
                         linestyle='dotted',
                         linewidth=0.5,
                         label='Band Boundaries')
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
        axs[1][0].vlines(x=boundaries,
                         ymin=min(np.amin(np.abs(m_xy_slr)), np.amin(np.abs(m_xy_slfrank))),
                         ymax=max(np.amax(np.abs(m_xy_slr)), np.amax(np.abs(m_xy_slfrank))),
                         colors='gray',
                         linestyle='dotted',
                         linewidth=0.5,
                         label='Band Boundaries')
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
        axs[0][1].vlines(x=boundaries,
                         ymin=min(np.amin(m_z_slr), np.amin(m_z_slfrank)),
                         ymax=max(np.amax(m_z_slr), np.amax(m_z_slfrank)),
                         colors='gray',
                         linestyle='dotted',
                         linewidth=0.5,
                         label='Band Boundaries')
        axs[0][1].plot(omega, m_z_slr,
                       linewidth=0.5,
                       label=r'SLR',
                       color='tab:orange')
        axs[0][1].plot(omega, m_z_slfrank,
                       linewidth=0.5,
                       label=r'SLfRank',
                       color='tab:blue')

    return fig
