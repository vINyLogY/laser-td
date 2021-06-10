#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
import sys
from builtins import zip

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from minitn.lib.units import Quantity
from scipy import integrate
from tqdm import tqdm

rc('font', family='Times New Roman')
rc('text', usetex=True)

os.chdir(os.path.abspath(sys.path[0]))

DTYPE = np.complex128
NUM = int(1e5)

# Parameters definitions
omega = Quantity(1, 'eV').value_in_au
omega1 = 2.0 * omega
omega2 = omega
# alpha = mu1 / mu0
mu0 = 1.0  # e * nm
alpha = 0.1
e1 = Quantity(2.3 * mu0, 'eV').value_in_au  # V / nm
e2 = Quantity(2.3 * mu0, 'eV').value_in_au  # V / nm
gamma1 = 1.0 / Quantity(2, 'fs').value_in_au
gamma2 = 1.0 / Quantity(2, 'fs').value_in_au
phi1, phi2 = 0, 0
t1, t2 = 0, 0
prefix = 'us-laser'


def gaussian_envelop(gamma, t):
    return np.exp(-gamma**2 * t**2)


def light_field(t):
    l1 = e1 * gaussian_envelop(gamma1, t - t1) * np.cos(omega1 * (t - t1) + phi1)
    l2 = e2 * gaussian_envelop(gamma2, t - t2) * np.cos(omega2 * (t - t2) + phi2)
    return l1 + l2


def func(t, x):
    h02 = np.exp(-2.0j * omega * t) * light_field(t)
    h12 = alpha * light_field(t)
    hamiltonian = -1.0j * np.array(
        [[0, 0, h02], 
         [0, 0, h12], 
         [np.conj(h02), np.conj(h12), 0]],
        dtype=DTYPE
    )
    return np.dot(hamiltonian, x)


def pop_plotter(t, b0, b1, b2, lat):
    t_ = np.array([Quantity(ti).convert_to('fs').value for ti in t])
    fig, ax1 = plt.subplots()
    ax1.plot(t_, np.abs(b0)**2, label=r"""$\langle 0 | 0 \rangle$""")
    ax1.plot(t_, np.abs(b1)**2, label=r"""$\langle 1 | 1 \rangle$""")
    ax1.plot(t_, np.abs(b2)**2, label=r"""$\langle 2 | 2 \rangle$""")
    ax1.set_ylabel('Population')
    ax1.set_xlabel('Time (fs)')
    ax1.set_xlim(-100, 100)

    ax2 = ax1.twinx()
    ax2.plot(
        t_,
        [Quantity(light_field(t_i) / mu0).convert_to('eV').value
         for t_i in t], 'k--')
    ax2.set_ylabel('Laser Strength (V / nm)')

    fig.legend(loc="upper left")
    fig.savefig('{}-{:04.1f}fs-population.pdf'.format(prefix, lat))
    plt.close(fig)
    return


def coh_plotter(t, b0, b1, b2, lat):
    t_ = np.array([Quantity(ti).convert_to('fs').value for ti in t])
    fig, ax1 = plt.subplots()
    ax1.plot(t_, np.imag(b1 * b2), label=r"""Im $\langle 1 | 2 \rangle$""")
    ax1.set_ylabel('Coherence')
    ax1.set_xlabel('Time (fs)')
    ax1.set_xlim(-100, 100)

    ax2 = ax1.twinx()
    ax2.plot(
        t_,
        [Quantity(light_field(t_i) / mu0).convert_to('eV').value
         for t_i in t], 'k--')
    ax2.set_ylabel('Laser Strength (V / nm)')

    fig.legend(loc="upper left")
    fig.savefig('{}-{:04.1f}fs-coherence.pdf'.format(prefix, lat))
    plt.close(fig)
    return

def end_plotter(dt, b0, b1, b2):
    plt.plot(dt, np.real(np.conj(b2) * b1), '-.', label=r"""Re $\rho_{12}$""")
    plt.plot(dt, np.imag(np.conj(b2) * b1), label=r"""Im $\rho_{12}$""")
    plt.ylabel('Coherence')
    plt.xlabel('Latency (fs)')
    plt.xlim(-80, 80)

    plt.legend(loc="best")
    plt.savefig('{}-end.pdf'.format(prefix))
    return



def main(l1, l2):
    global t2, omega1, omega2, prefix
    t0 = Quantity(1, 'ps').value_in_au
    x0 = np.array([1., 0., 0.], dtype=DTYPE)
    t_space = np.linspace(-t0, t0, num=NUM)

    if l1 == 1:
        omega1 = 2.0 * omega
    elif l1 == 2:
        omega1 = omega
    else:
        raise RuntimeError('argv 1 not valid: must be 1 or 2.')
    if l2 == 1:
        omega2 = 2.0 * omega
    elif l2 == 2:
        omega2 = omega
    else:
        raise RuntimeError('argv 2 not valid: must be 1 or 2.')
    prefix = 'us-laser{}{}'.format(l1, l2)
    print('Task: {}'.format(prefix))

    data = []
    for latency in tqdm(range(-200, 200), desc=prefix):
        latency = 0.4 * latency     
        t2 = Quantity(latency, 'fs').value_in_au
        
        solver = integrate.solve_ivp(func, (-t0, t0),
                                    x0,
                                    t_eval=t_space,
                                    method='RK45',
                                    max_step=2.0 * t0 / NUM)
        t = solver.t
        b0, b1, b2 = solver.y
        # np.savetxt('{}-td_wfn-{}fs.dat'.format(prefix, latency),
        #           list(zip(t, b0, b1, b2)),
        #           header='t, b0, b1, b2')
        data.append((latency, b0[-1], b1[-1], b2[-1]))
        # if int(latency * 10) % 100 == 0:
            # coh_plotter(t, b0, b1, b2, latency)
            # pop_plotter(t, b0, b1, b2, latency)

    data = np.array(data)
    np.savetxt('{}-end.dat'.format(prefix, latency), data, header='dt, b0, b1, b2')
    end_plotter(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
    return


if __name__ == "__main__":
    l1, l2 = int(sys.argv[1]), int(sys.argv[2])
    main(l1, l2)
    #prefix = 'laser21'
    #data = np.loadtxt('{}-end.dat'.format(prefix), dtype=np.complex128)
    #end_plotter(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
    
