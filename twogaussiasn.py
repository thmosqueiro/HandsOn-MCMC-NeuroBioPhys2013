import random,math
import numpy as np
from chaco.shell import *
import matplotlib.pyplot as plt


def generate2gaussians(N = 100, mu1 = 0., mu2 = 1., sigma1 = .4, \
                sigma2 = .4, mix = 0.5):
    
    s = np.array([])
    for i in range(N):
        if random.uniform(0,1) > mix:
            s = np.append(s, random.gauss(mu1, sigma1) )
        else:        
            s = np.append(s, random.gauss(mu2, sigma2) )
            
    return s



s = generate2gaussians(N=2000, mu1=-1., mu2=1.0, mix = 0.0)

hist, bin_edges = np.histogram(s, density=True, bins=15)
plt.bar(bin_edges[:-1], hist, width=0.2)
plt.show()

np.savetxt('dados_nonmixture.data', s)




s = generate2gaussians(N=2000, mu1=-1., mu2=1.0, mix = 0.25)

hist, bin_edges = np.histogram(s, density=True, bins=15)
plt.bar(bin_edges[:-1], hist, width=0.2)
plt.show()

np.savetxt('dados_mixture.data', s)