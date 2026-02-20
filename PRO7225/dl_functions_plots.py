"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 06 - Just a few plot functions to make graphs for the slides and report
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh

# 1 - Activation functions ==================

# ReLU
def ReLU(x):
    return np.maximum(0,x)

# Sigmoid
def Sigmoid(x):
    return 1/(1+np.exp(-x))

# 2 - Plots ==================================

x = np.linspace(-10,10,1000)

# # Plot configuration
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6, 4))
# # Curves
# ax.plot(x, ReLU(x), color='purple')
# # Others
# ax.grid(lw=0.2, color='gray')
# plt.show()

# # Plot configuration
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=(6, 4))
# # Curves
# ax.plot(x, Sigmoid(x), color='red')
# # Others
# ax.grid(lw=0.2, color='gray')
# plt.show()

# 3 - Testing fading models ================

x = np.linspace(0.1,20,1000)

friis = 20*np.log10(x)
noise = np.random.rayleigh(scale=1.0, size=len(x))
factor = np.random.randn(1000)

plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x, friis, label='Friis equation')
ax.plot(x, friis + factor*noise, label='With shadowing + fading', alpha=0.6)
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Path loss [dB]')
ax.set_title('Effects on the path loss calculation')
ax.legend()
plt.show()


