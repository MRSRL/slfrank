"""
script for using slfrank to create slr pulse with set parameters and export with rfpf to interface
eg. in jstmc or emc
_________________
Jochen Schmidt
15.02.2023
"""

import numpy as np
import matplotlib.pyplot as plt
import slfrank
import pathlib as plib
from rf_pulse_files import rfpf
from slr_creation import options

import logging


def main(opts: options.Config):
    opts.display()
    gamma_Hz = 42577478.518
    bandwidth = gamma_Hz * opts.desiredSliceGrad * 1e-3 * opts.desiredSliceThickness * 1e-3
    if opts.desiredDuration > 0:
        opts.timeBandwidth = bandwidth * opts.desiredDuration * 1e-6
    else:
        opts.desiredDuration = int(opts.timeBandwidth / bandwidth * 1e6)

    opts.display()
    # init object
    slr_pulse = rfpf.RF(
        name="slfrank_lin_phase_refocus",
        duration_in_us=opts.desiredDuration,
        bandwidth_in_Hz=bandwidth,
        time_bandwidth=opts.timeBandwidth,
        num_samples=opts.numSamples
    )

    # set solver
    solver = 'PDHG'
    pulse_type = opts.pulseType
    phase_type = opts.phaseType

    logging.info(f"Generating pulse"
                 f"\t\t__type: {pulse_type} \t __phase type {phase_type}")

    # getting length n complex array
    pulse_slfrank = slfrank.design_rf(
        n=opts.numSamples, tb=opts.timeBandwidth,
        ptype=pulse_type, phase=phase_type,
        d1=opts.rippleSizes, d2=opts.rippleSizes,
        solver=solver, max_iter=opts.maxIter)

    slr_pulse.amplitude = np.real(pulse_slfrank)
    slr_pulse.phase = np.angle(pulse_slfrank)

    logging.info(f'SLfRank:\tEnergy={np.sum(np.abs(pulse_slfrank)**2)}\tPeak={np.abs(pulse_slfrank).max()}')

    logging.info("plotting")
    fig = slfrank.plot_slr_pulses(
            np.full_like(pulse_slfrank, np.nan, dtype=complex),
            pulse_slfrank, ptype=pulse_type, phase=phase_type,
            omega_range=[-1, 1], tb=opts.timeBandwidth, d1=opts.rippleSizes, d2=opts.rippleSizes)
    plt.tight_layout()
    plt.show()

    out_path = plib.Path(opts.outputPath).absolute()
    plot_file = out_path.joinpath(f'{pulse_type}_{phase_type}.png')
    pulse_file = out_path.joinpath(f"slfrank_{pulse_type}_pulse_{phase_type}_phase.pkl")

    logging.info(f"saving plot {plot_file}")
    fig.savefig(plot_file, bbox_inches="tight", transparent=True)

    logging.info(f"saving pulse {pulse_file}")
    slr_pulse.save(pulse_file)

    logging.info(f"finished")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    # get cmd line input
    parser, args = options.createCommandlineParser()

    logging.info("set parameters")
    conf_opts = options.Config.from_cmd_args(args)

    try:
        main(conf_opts)
    except Exception as e:
        logging.error(e)
        parser.print_usage()
