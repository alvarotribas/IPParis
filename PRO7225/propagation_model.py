"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 02 - Modelling antenna radiation pattern and propagation

Author: Alvaro RIBAS
"""

# 0 - Imports =============================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.special import i0 # Modified Bessel function Ia of order a=0
from scipy.stats import vonmises
import math

# 1 - Directive antenna model ===============================================================================================

# 1.1 - Distribution models

# cos^m(azimuth) model, from Matlab
def cosine8(x):
    return (np.cos(x))**8
# Normalized von Mises distribution formula
def vonMises(x, mu, kappa):
    vm = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
    return vm/max(vm)
# Cardioid distribution
def cardioid(l, theta, phase):
    return l + l*np.cos(theta + phase)

# 1.2 - Paramaters

# Angle
theta = np.arange(-np.pi, np.pi, 1e-3)
# Mean and dispersion of von Mises
mu, kappa = 0, 6 # Omnidirectional: mu = kappa = 0
# Length and phase of the cardioid
l, phase = 0.5, 0

# Calling
r1 = cosine8(theta)
r2 = vonMises(theta, mu, kappa)
r3 = cardioid(l, theta, phase)
# -3 dB limit

# 2.4 - Plots from antenna patterns
# When plotting, choose one of the polar 2 configurations. The plots are different for LTE and 5G

# Plot configuration
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
fig = plt.figure(figsize=(12, 6))

# Polar 1
ax1 = plt.subplot(1, 2, 1, projection='polar')
ax1.plot(theta, r1, label=r'$\cos^8(\theta)$', color='red')
ax1.plot(theta, r2, label=r'vonMises($\theta, \mu=0, \kappa=7$)', color='green')
ax1.plot(theta, r3, label=r'Cardioid($l=0.5, \theta$)', color='blue')
ax1.set_title('Antenna patterns of different distribution models')
ax1.legend(loc='lower left')

# # Polar 2 for LTE case
# ax2 = plt.subplot(1, 2, 2, projection='polar')
# angle = 120
# phase = [0, np.pi*(angle/180), -np.pi*(angle/180)]
# for p in phase:
#     card = cardioid(l, theta, p)
#     ax2.plot(theta, card, label=rf'$\phi={round(p*180/(np.pi))}$', color='blue')
# ax2.plot(theta, [0.5]*len(theta), color='black', label='-3 dB limit')
# ax2.set_title('Cardioid patterns for various $\phi$ (LTE)')
# ax2.legend(loc='lower left')  # moves legend outside

# Polar 2 for 5G NR case
ax2 = plt.subplot(1, 2, 2, projection='polar')
mu = [1, 3, 5]
for m in mu:
    von_mises = vonMises(theta, m, kappa)
    ax2.plot(theta, von_mises, label=rf'$\mu={m}$')
ax2.plot(theta, [0.5]*len(theta), color='black', label='-3 dB limit')
ax2.set_title('Normalized von Mises patterns for $\kappa = 2$ (5G NR)')
ax2.legend(loc='lower left')  # moves legend outside

# Final plots
plt.tight_layout()
plt.show()

# 2 - Propagation model: Physical and classification losses ======================================================================
# This section was inspired by the Liu et al. "PINN and GNN-based RF Map Construction" paper

# 2.1 - Power constraint of the physical model

# Free-space path loss (FSPL) Model
def FSPL(d, f):
    return 20*np.log10(d) + 20*np.log10(f) + 20*np.log10(4*np.pi/c)

# Parameters
freqs_orange = [3710e6, 2635e6, 2155e6, 1805e6, 935e6, 811e6, 763e6] # Fixed frequency [MHz] examples for different Orange bands
d = np.linspace(0,410,10000) # Distance [m]

# 2.2 - Plots from propagation model

# Plot configuration
fig, ax = plt.subplots(figsize=(8, 4))
# Curves
for f in freqs_orange:
    ax.plot(d, FSPL(d,f), label=f'{f/1e6:.0f} MHz')
ax.set_ylabel('Path loss [dB]')
ax.set_xlabel('Distance [m]')
# Others
ax.set_title('Examples of Orange frequency bands FSPLs')
ax.grid(lw=0.2, color='gray')
ax.legend()
plt.show()
