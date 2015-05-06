__author__ = 'tiago'

import matplotlib.pyplot as plt
import numpy as np

pca_explained_variance = [  8.03567024e-01, 1.24306703e-01, 3.20190115e-02, 1.60813930e-02
, 9.71917145e-03, 4.05066977e-03, 2.91286160e-03, 2.11078484e-03
, 1.30244633e-03, 1.17795208e-03, 7.74285182e-04, 5.29918455e-04
, 3.38717569e-04, 2.47036545e-04, 1.63704424e-04, 1.09402628e-04
, 5.80498174e-05, 5.45211471e-05, 4.49878768e-05, 3.69859899e-05
, 3.47807139e-05, 2.48923812e-05, 2.12581693e-05, 1.84579799e-05
, 1.67236179e-05, 1.65034587e-05, 1.34758840e-05, 1.31249187e-05
, 1.26078527e-05, 1.12681560e-05]

plt.figure(figsize=(4,3))
plt.bar(np.arange(len(pca_explained_variance)), pca_explained_variance)
plt.xlabel('PC')
plt.ylabel('% explained variance')
plt.tight_layout()
plt.savefig('plots/SI_pca.pdf')