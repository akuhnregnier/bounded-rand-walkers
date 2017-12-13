#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:41:00 2017

@author: luca
"""

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import os
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif', size = 15)

plot_dir = 'plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

dl = 1.
ndiv = int(40/dl)
colormark = 'k'

@njit
def random_walk1Djit(N,dl):
    steps = [np.random.random()]
    for i in range(N+1):

        x = steps[-1]

        if x+dl >= 1:
            highx = 1
        else:
            highx = x+dl


        if x-dl <= 0:
            lowx = 0
        else:
            lowx = x-dl

        steps.append(np.random.uniform(lowx ,highx))
    return steps

a = np.array(random_walk1Djit(int(3e7),dl))
print(len(a))
print('Data generated correctly')

steps = a[1:] - a[:-1]

xdata1=np.abs(steps[:-1])
ydata1=steps[1:]*steps[:-1]

xdata2=np.abs(steps[:-2])
ydata2=steps[2:]*steps[:-2]

xdata3=np.abs(steps[:-3])
ydata3=steps[3:]*steps[:-3]

#one-timestep correlation
plt.figure(1)
plt.hist2d(xdata1,ydata1,bins=ndiv,norm=LogNorm(),
           normed=True)

'''
#two-timesteps correlation
plt.figure(2)
plt.hist2d(xdata2,ydata2,bins=ndiv,norm=LogNorm())

#three-timesteps correlation
plt.figure(3)
plt.hist2d(xdata3,ydata3,bins=ndiv,norm=LogNorm())
'''
binedges = np.linspace(0,1,ndiv)
bincenters = (binedges[1:] + binedges[:-1])/2.

@njit
def looper(xdata,ydata,ndiv,binedges):
    means = np.zeros(ndiv-1)
    index = 0
    for i in range(ndiv):

        averager = 0.
        counter = 0

        for j in range(len(xdata)):
            if binedges[i] < xdata[j] < binedges[i+1]:
                averager += ydata[j]
                counter += 1
        if counter != 0:
            means[index] = averager/counter
            index += 1
    return means, index

means1, index1 = looper(xdata1,ydata1,ndiv,binedges)
means2, index2 = looper(xdata2,ydata2,ndiv,binedges)
means3, index3 = looper(xdata3,ydata3,ndiv,binedges)

plt.figure(1)
# plt.title('One-timestep correlation')
plt.plot(bincenters, means1)
plt.xlabel(r'$|d_i|$')
plt.ylabel(r'$d_i * d_{i-1}$')
plt.colorbar(label=r'$p$')
plt.savefig(os.path.join(plot_dir,
                         'one-timestep-correlation-{:}.png'.format(dl)),
            dpi=600, bbox_inches='tight')

'''
plt.figure(2)
plt.title('Two-timesteps correlation')
plt.plot(bincenters,means2)

plt.figure(3)
plt.title('Three-timesteps correlation')
plt.plot(bincenters,means3)

#comparisons
plt.figure(4)
plt.title('Comparison correlations at fixed $\Delta \ell$')
plt.plot(bincenters/dl,np.log(-means1),'v',markerfacecolor='none',markeredgecolor=colormark)
plt.plot(bincenters/dl,np.log(-means2),'s',markerfacecolor='none',markeredgecolor=colormark)
plt.plot(bincenters/dl,np.log(-means3),'D',markerfacecolor='none',markeredgecolor=colormark)
plt.xlabel('$|d_i|/\Delta \ell$')
plt.ylabel('$d_{i+1} d_i$')
plt.xlim((0,1))
'''


'''
if dl == 0.2:
    fig , axis = plt.subplots(1,2,squeeze=True,sharey = True)
    axis[0].plot(bincenters[:index1]/dl,np.log(-means1[:index1]),'v',markerfacecolor='none',markeredgecolor=colormark, label='One-Timestep')
    axis[0].plot(bincenters[:index2]/dl,np.log(-means2[:index2]),'s',markerfacecolor='none',markeredgecolor=colormark, label='Two-Timestep')
    axis[0].plot(bincenters[:index3]/dl,np.log(-means3[:index3]),'D',markerfacecolor='none',markeredgecolor=colormark, label='Three-Timestep')
    axis[0].legend()


if dl == 0.8:
    #fig , axis = plt.subplots(1,2,squeeze=True,sharey = True)
    plt.figure(1)
    axis[1].plot(bincenters[:index1]/dl,np.log(-means1[:index1]),'v',markerfacecolor='none',markeredgecolor=colormark)
    axis[1].plot(bincenters[:index2]/dl,np.log(-means2[:index2]),'s',markerfacecolor='none',markeredgecolor=colormark)
    axis[1].plot(bincenters[:index3]/dl,np.log(-means3[:index3]),'D',markerfacecolor='none',markeredgecolor=colormark)

    axis[0].set_xlabel('$|d_i|/ \Delta \ell$')
    axis[1].set_xlabel('$|d_i|/ \Delta \ell$')
    axis[0].set_ylabel(r'$ \log \left( - \langle d_{i} \times d_{i-1} \rangle \right) $')
'''
