import math
import numpy as np
#from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg
#from numpy import linalg
"""
this script defines a system with two DoFs (typical section, with pitch and plunge mpotion)
defines the usual mass, damping and stiffness matrices with two different formulations of the aerodynamic forces:
quasi-steady and unsteady (Theodorsen)
subsequently, it calculates and plots the real and imaginary parts of the dynamic response to a constant (in frequency) forcing term in "z" 
"""


m = 75.
c = 0.25 #0.45
e = 0.15 * c # positive ahead of EA
I = m * (0.25 * c) ** 2
xcg = -0.05 * c # positive ahead of EA

cla = 2 * np.pi

#omega_plunge = 3.30 * 2 * np.pi
#omega_pitch = 5.20 * 2 * np.pi

omega_plunge = 13.30 * 2 * np.pi
omega_pitch = 15.20 * 2 * np.pi

kz = m * omega_plunge ** 2
kt = I * omega_pitch ** 2

M = np.array([[m, m * xcg], [m * xcg, I + m * xcg ** 2]])
M_1 = linalg.inv(M)
Ks = np.diag([kz, kt])
Ka = cla * c * np.array([[0, 1.], [0., e]]) 
Cs = 0.001 * 2 * np.sqrt(np.diag(M) *np.diag(Ks)) #np.zeros((2, 2))
Ca0 = np.zeros((2, 2))
Ca1 = c * cla * np.array([[-1., (e + 0.25 * c)], [-e, e * (e + 0.25 * c) - c ** 2 / 16]]) # I used one of the approximated formulations available in the literature 

rho = 1.22
v = 80.
q = 0.5 * rho * v ** 2

omega_min = 0.1
omega_max = 150. # RAD/s
n_omega = 400
omega = np.linspace(omega_min, omega_max, n_omega)
k = omega * c / (2 * v)

# Jones coeffs
A1 = 0.335
A2 = 0.165
b1 = 0.30
b2 = 0.0455
j = np.sqrt(-1 + 0j)
Ck = 1 - A1/(1 - b1/(j*k)) - A2/(1 - b2/(j*k))

x = np.zeros((2, n_omega), dtype=complex)
x2 = np.zeros((2, n_omega), dtype=complex)
# f = np.array([0., 100]) # plunge 
f = np.array([0., 100]) # pitch

for iomega in range(n_omega):
    Hael = -omega[iomega] ** 2 * M +  Ks - q * Ka + j * omega[iomega] * (Cs - 0.5 * rho * v *Ca1) 
    x[:, iomega] = linalg.inv(Hael) @ f
    Hael = -omega[iomega] ** 2 * M +  Ks -  Ck[iomega] * q * Ka + j * omega[iomega] * (Cs - Ck[iomega] * 0.5 * rho * v *Ca1) 
    x2[:, iomega] = linalg.inv(Hael) @ f

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 5.25)
ax[0].plot(k, np.abs(x[0, :]), 'b-', linewidth=3, label='QS')
ax[0].plot(k, np.abs(x2[0, :]), 'r-', linewidth=3, label='UNSTEADY')
ax[0].set_xlabel(r'$k\,[\,]$')        
ax[0].set_ylabel(r'$\|z\|\,[m]$') 
ax[0].legend()
ax[0].grid()
ax[1].plot(k, np.abs(x[1, :]), 'b-', linewidth=3, label='QS')
ax[1].plot(k, np.abs(x2[1, :]), 'r-', linewidth=3, label='UNSTEADY')
ax[1].set_xlabel(r'$k\,[\,]$')        
ax[1].set_ylabel(r'$\|\theta\|,[RAD]$') 
ax[1].grid()
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 5.25)
ax[0].plot(k, np.angle(x[0, :]), 'b-', linewidth=3, label='QS')
ax[0].plot(k, np.angle(x2[0, :]), 'r-', linewidth=3, label='UNSTEADY')
ax[0].set_xlabel(r'$k\,[\,]$')        
ax[0].set_ylabel(r'$PH(z)\,[RAD]$') 
ax[0].legend()
ax[0].grid()
ax[1].plot(k, np.angle(x[1, :]), 'b-', linewidth=3, label='QS')
ax[1].plot(k, np.angle(x2[1, :]), 'r-', linewidth=3, label='UNSTEADY')
ax[1].set_xlabel(r'$k\,[\,]$')        
ax[1].set_ylabel(r'$PH(\theta),[RAD]$') 
ax[1].grid()
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 5.25)
ax[0].plot(k, x[0, :].real, 'b-', linewidth=3, label='QS')
ax[0].plot(k, x2[0, :].real, 'r-', linewidth=3, label='UNSTEADY')
ax[0].set_xlabel(r'$k\,[\,]$')        
ax[0].set_ylabel(r'$Re(z)\,[m]$') 
ax[0].legend()
ax[0].grid()
ax[1].plot(k, x[0, :].imag, 'b-', linewidth=3, label='QS')
ax[1].plot(k, x2[0, :].imag, 'r-', linewidth=3, label='UNSTEADY')
ax[1].set_xlabel(r'$k\,[\,]$')        
ax[1].set_ylabel(r'$Im(z),[m]$') 
ax[1].grid()
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 5.25)
ax[0].plot(k, x[1, :].real, 'b-', linewidth=3, label='QS')
ax[0].plot(k, x2[1, :].real, 'r-', linewidth=3, label='UNSTEADY')
ax[0].set_xlabel(r'$k\,[\,]$')        
ax[0].set_ylabel(r'$Re(\theta)\,[RAD]$') 
ax[0].legend()
ax[0].grid()
ax[1].plot(k, x[1, :].imag, 'b-', linewidth=3, label='QS')
ax[1].plot(k, x2[1, :].imag, 'r-', linewidth=3, label='UNSTEADY')
ax[1].set_xlabel(r'$k\,[\,]$')        
ax[1].set_ylabel(r'$Im(\theta),[RAD]$') 
ax[1].grid()
ax[1].legend()
plt.show()