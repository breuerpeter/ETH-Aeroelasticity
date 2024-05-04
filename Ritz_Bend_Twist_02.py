import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

"""
the script models the struct. and aerodynamic stiffness matrices of a straight wing
taking sweep angle and shear centre into account
it calculates q_divergence and the response to a constant aerodynamic load

by varying the shear centre position and / or the sweep angle we (should) notice that the
twisting decreases and q_divergence increases until it becomes a complex number (indicating that the system does not experience any divergence)
"""
cla = 2*np.pi
chord = 1.
cm0 = 0.01
cl0 = 0.10 
e = 0.2*chord
eb = -0.1*chord
alpha_0 = 2.0*np.pi/180.
Lambda =  30.01*np.pi/180.
tanL = np.tan(Lambda)
span = 10. 

GJ = 200000.
EI = 5000000.

n_fine = 320
y = np.linspace(0.,span,n_fine)
tip_loss_fine = np.sqrt(1-(y/span)**2)

y = np.linspace(0.,1,n_fine)
ydim = y * span
tip_loss_fine = np.sqrt(1-(y)**2)

phi1 = 2 * y -  y ** 2
dphi1_dy = (2.0 - 2 * y) 
d2phi1_dy2 = (2.0 + 0. * y) 

psi1 = 8 * (1 / 4 * y ** 2 -1 / 6 * y **3 + 1 / 24 * y ** 4) 
dpsi1_dy = 8 * (y / 2 - 1 / 2 * y ** 2 + 1 / 6 * y ** 3)  
d2psi1_dy2 = 8 * (1 / 2 - y + 1 / 2 * y ** 2) 
d3psi1_dy3 = -8 * (- 1. + y ) # signa change to account for coordinate system

ks11 = GJ * np.trapz(dphi1_dy*dphi1_dy, y) / (span)
ks22 = EI * np.trapz(d2psi1_dy2*d2psi1_dy2, y) / (span ** 3)
ks12 = eb * EI * np.trapz(d3psi1_dy3*dphi1_dy, y) / (span ** 3)

rho = 1.22
v = 50.0 
q = 0.5 * rho * v ** 2

ka11 =  chord * e * cla * np.trapz(phi1*phi1, y) *span
ka21 =  chord * cla * np.trapz(psi1*phi1, y) *span
ka12 = - chord * e * cla * tanL * np.trapz(dpsi1_dy*phi1, y)
ka22 = - chord * cla * tanL * np.trapz(dpsi1_dy*psi1, y) 

load1 =  q * chord * e * cla * alpha_0 * np.trapz(phi1, ydim)
load2 =  q * chord * cla * alpha_0 * np.trapz(psi1, ydim)

ks = np.array([[ks11, ks12], [ks12, ks22]])

ka = np.array([[ka11, ka12], [ka21, ka22]])

loads = np.array([load1, load2])

a = linalg.inv(ks-q*ka)@loads

qd, rd = linalg.eig(linalg.inv(ka) @ ks)


print(a)


#print(qd)
#print(q*ka)
#print(ks)

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(9.25, 5.25)
ax[0].plot(y,a[0]*phi1, 'k-')
# ax[0].set_xlabel(r'$y\,[m]$')        
ax[0].set_ylabel(r'$\theta\,[RAD]$')        
ax[0].grid()
ax[1].plot(y,a[1]*psi1, 'k-')
ax[1].set_xlabel(r'$y\,[m]$')        
ax[1].set_ylabel(r'$w\,[m]$')        
ax[1].grid()
plt.show()