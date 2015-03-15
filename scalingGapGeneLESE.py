# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:22:42 2015

@author: Eric

Load and analyze gap gene long/short data
"""


import numpy as np
import os
import scipy.io as sio

import config
reload(config)
import scalingBicoidFinalReally
reload(scalingBicoidFinalReally)



if __name__ == '__main__':
    try:
        (results, ks_results) = np.load(os.path.join(config.tmp_path, 'results.npy'))
    except IOError:
        results = []
        ks_results = []


        """ Load gap gene LE&SE data """
        dd = sio.loadmat(os.path.join(config.scaling_data_path,
                                      'ScalingData1And23.mat'), squeeze_me=True)
        dat = dd['RawData']['M'].item()
        profiles_symmetric = []
        profiles_ventral = []
        profiles_dorsal = []
        len_symmetric = []
        len_ventral = []
        len_dorsal = []
        i_session = 0;
        # Index of gap gene staining session
        gap_gene_name_l = ['Kni', 'Kr', 'Gt', 'Hb']
        # Index of channels corresponding to gap genes: from dd['RawData']['M']
        
        # Read data into dict
        for l_gap_gene in xrange(gap_gene_name_l):
            for p, or_v, le_v, ag_v in zip(dat['Em'][i_session]['Profile'],
                                           dat['Em'][i_session]['orientation'],
                                           dat['Em'][i_session]['EL'],
                                           dat['Em'][1]['Emage']):
                # {{{continue on with this, using the code from scalingBicoidFinalReally as a template: but first, ask Tiago if he's already written this code, to save time doing whatever clipping/smoothing, etc. procedure he did with Bcd LE/SE}}}