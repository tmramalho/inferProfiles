# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:27:16 2015

@author: Eric
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import scipy.stats as stats

import config
reload(config)


if __name__ == '__main__':

    # Load data
    if 'dat' not in dir():
        dd = sio.loadmat(os.path.join(config.scaling_data_path,
                                         'ScalingData1And23.mat'), squeeze_me=True)
        dat = dd['RawData']['M'].item()
        del dd

    # Extract stats
    length_l = [em['EL'] for em in dat[0]['Em']] + [em['EL'] for em in dat[1]['Em']]
    width_l = [em['H'] for em in dat[0]['Em']] + [em['H'] for em in dat[1]['Em']]
    volume_l = [4.0*np.pi/3 * (length/2.0) * (width/2.0)**2 for length, width in zip(length_l, width_l)]

    # Plot
    fig = plt.figure(figsize=(18, 6))

    ax = fig.add_subplot(1, 3, 1)
    plt.scatter(length_l, width_l)
    ax.set_xlabel('Embryo length ($\mu$m)')
    ax.set_ylabel('Embryo width ($\mu$m)')

    ax = fig.add_subplot(1, 3, 2)
    plt.scatter(length_l, volume_l)
    ax.set_xlabel('Embryo length ($\mu$m)')
    ax.set_ylabel('Embryo volume ($\mathrm{\mu m}^3$)')

    ax = fig.add_subplot(1, 3, 3)
    plt.scatter(np.log(length_l), np.log(volume_l))
    slope, intercept, r, __, __ = stats.linregress(np.log(length_l), np.log(volume_l))
    xlim_a = np.array(ax.get_xlim())
    plt.plot(xlim_a, slope*xlim_a+intercept, 'r')
    ax.set_xlabel('ln(Embryo length)')
    ax.set_ylabel('ln(Embryo volume)')
    ax.set_title('V = L^{:1.2f}, r = {:0.2f}'.format(slope, r))