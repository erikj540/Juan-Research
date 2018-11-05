# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:38:24 2018

@author: Erik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import timeit

#%%
np.random.seed(seed=1)
#%%
N = 10**3
delta = 0.64
K = 0.885
y0 = [(np.random.uniform(low=-0.1, high=0.1) + np.random.uniform(low=-0.1, high=0.1)*np.complex(0,1)) for ii in range(0,N)]
t0 = 0

interval = [-np.pi*delta/2, np.pi*delta/2]
#interval = [0,1]
omegas = [interval[0] + ii*(interval[1]-interval[0])/(N-1) for ii in range(0,N)]

def f(t, y):
    z_bar = np.sum(y)/N
    
    return [y[n]*(1 - np.abs(y[n])**2 + np.complex(0,1)*omegas[n]) + K*(z_bar - y[n]) for n in range(0,len(y))]


#%%
solver = ode(f).set_integrator('zvode')
solver.set_initial_value(y0, t0)

#%%
t1 = 200
dt = .05
time = []
r_modulus = []
r_angle = []

start = timeit.timeit()

while solver.successful() and solver.t < t1:
    time.append(solver.t)
    r = np.sum(solver.y)/N 
    r_modulus.append(np.abs(r))
    r_angle.append(np.angle(r))
#    print(solver.t)
    solver.integrate(solver.t+dt)

end = timeit.timeit()
#%%
plt.plot(time, r_modulus) 
plt.xlabel('t')
plt.ylabel('r(t)')

#%%
plt.plot(time, r_angle) 
plt.xlabel('t')
plt.ylabel('Theta')

#%%
plt.plot(time, np.sin(r_modulus)) 
plt.xlabel('t')
plt.ylabel('Sin(Theta)')