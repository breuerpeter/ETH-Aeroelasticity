import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg

plotshape = False

l = 1.0
m = 10.0
EI = 2e1
rho = 0.15
Mach = 2.0

ndof = 24

nx = 200
x = np.linspace(0, l, nx)

thickness = 1.25 - 0.65 * np.sin(np.pi*x/l)

phi = np.zeros((ndof, nx))

for imode in range(ndof):
    phi[imode, :] = np.sin(np.pi*x/(l/(imode+1)))

phi_x = np.gradient(phi, x, axis=1)

phi_xx = np.gradient(phi_x, x, axis=1)

if plotshape: 
    fig, ax = plt.subplots()
    for imode in range(ndof):
        ax.plot(x, phi[imode, :],label='Shape F. '+str(imode), linewidth=2)
    ax.set_xlabel(r'$x\,[m]$')
    ax.set_ylabel(r'$z\,[m]$')
    ax.grid(visible=True)
    ax.legend()
    plt.show()    
    
    fig, ax = plt.subplots()
    for imode in range(ndof):
        ax.plot(x, phi_x[imode, :],label='Shape F./x '+str(imode), linewidth=2)
    ax.set_xlabel(r'$x\,[m]$')
    ax.set_ylabel(r'$z\,[m]$')
    ax.grid(visible=True)
    ax.legend()
    plt.show()    

M = np.zeros((ndof, ndof))
K = np.zeros((ndof, ndof))
Ka = np.zeros((ndof, ndof))

for j in range(ndof):
    for i in range(ndof):
        M[i, j] = np.trapz(phi[i, :]*phi[j, :]*thickness, x)
        K[i, j] = np.trapz(phi_xx[i, :]*phi_xx[j, :]*thickness, x)
        Ka[i, j] = np.trapz(phi[i, :]*phi_x[j, :], x)
        
M = M * m * l
K = K * EI
Ka = Ka * rho / Mach

v_min = 0.1
v_max = 270.
n_vel = 120

v = np.linspace(v_min, v_max, n_vel)
ff = np.zeros((ndof, n_vel))
gg = np.zeros((ndof, n_vel))

for iv in range(n_vel):
    Kael = K + Ka * v[iv] ** 2
    l, r = linalg.eig(-linalg.inv(M) @ Kael)
    gg[:, iv] = np.sqrt(l).real
    ff[:, iv] = np.sqrt(l).imag
    
fig, ax = plt.subplots()
for imode in range(ndof):
    ax.plot(v, gg[imode, :],label='Eig. '+str(imode),linewidth=1)
ax.set_xlabel(r'$v\,[m/s]$')
ax.set_ylabel(r'$Re(\lambda)\,[RAD/s]$')
ax.grid(visible=True)
#ax.legend()
plt.show()

fig, ax = plt.subplots()
for imode in range(ndof):
    ax.plot(v, ff[imode, :],label='Eig. '+str(imode),linewidth=1)
ax.set_xlabel(r'$v\,[m/s]$')
ax.set_ylabel(r'$Im(\lambda)\,[RAD/s]$')
ax.grid(visible=True)
#ax.legend()
plt.show()

#print(np.sqrt(l))
#print(r)

modes = r.T @ phi

fig, ax = plt.subplots()
for imode in range(ndof):
    ax.plot(x, modes[imode, :],label='Mode '+str(imode), linewidth=1)
ax.set_xlabel(r'$x\,[m]$')
ax.set_ylabel(r'$z\,[m]$')
ax.grid(visible=True)
# ax.legend()
plt.show()

indices = np.argsort(np.abs(np.imag(np.sqrt(l))))
lll = [(0, ()), (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 9.25)
for imode in range(5):
    xi =  np.real(np.sqrt(l[indices[imode]])) / np.imag(np.sqrt(l[indices[imode]])) 
    ax[0].plot(x, modes[indices[imode], :],label='Mode '+str(imode+1) + r"$ \, \, \xi = $"+ str(xi)[:8] + r"$ \, \, \omega = $"+ str(np.imag(np.sqrt(l[indices[imode]])))[:8] + " RAD/s", linewidth=2,  color='k', linestyle= lll[imode])
ax[0].set_xlabel(r'$x\,[m]$')
ax[0].set_ylabel(r'$z\,[m]$')
ax[0].grid(visible=True)
ax[0].legend()

# no AERO
l0, r0 = linalg.eig(-linalg.inv(M) @ K)
indices = np.argsort(np.abs(np.imag(np.sqrt(l0))))
modes = r0.T @ phi

for imode in range(5):
    ax[1].plot(x, modes[indices[imode], :],label='Mode '+str(imode+1) + r"$ \, \, \omega = $"+ str(np.imag(np.sqrt(l0[indices[imode]])))[:8]  + " RAD/s", color='k', linewidth=2, linestyle= lll[imode])
ax[1].set_xlabel(r'$x\,[m]$')
ax[1].set_ylabel(r'$z\,[m]$')
ax[1].grid(visible=True)
ax[1].legend()
plt.show()
