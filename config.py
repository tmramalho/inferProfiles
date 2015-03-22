# -*- coding: utf-8 -*-
"""
Eric Smith, 2015-03-13

Provides for configuration settings for inferProfiles

Suffixes at the end of variable names (Eric's convention):
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s (or path): string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
"""


import socket


## Paths
hostname_s = socket.gethostname().lower()
erics_pcs_s_l = ['esmith11desktop', 'esmith13laptop']
erics_mac_s = 'esmith15laptop'
if hostname_s in erics_pcs_s_l:
    mutant_path = 'C:\\E\\Dropbox\\Lab\\MatlabData\\Staining\\Mariela_criticality'
    plots_path = 'C:\\E\\Dropbox\\Lab\\05_Scaling\\Plots\\inferProfiles\\tmp'
    scaling_data_path = 'C:\\E\\Dropbox\\Lab_Shared\\Scaling paper\\Data\\DataSets'
    tmp_path = 'C:\\E\\Dropbox\\Lab\\05_Scaling\\Results\\tmp'
elif hostname_s == erics_mac_s:
    mutant_path = '/Users/Eric/Dropbox/Lab/MatlabData/Staining/Mariela_criticality'
    plots_path = '/Users/Eric/Dropbox/Lab/05_Scaling/Plots/inferProfiles/tmp'
    scaling_data_path = '/Users/Eric/Dropbox/Lab_Shared/Scaling paper/Data/DataSets'
    tmp_path = '/Users/Eric/Dropbox/Lab/05_Scaling/Results/tmp'
else:
    mutant_path = 'data/criticality'
    plots_path = 'plots'
    scaling_data_path = 'data/scaling_data/DataSets'
    tmp_path = 'data/tmp'