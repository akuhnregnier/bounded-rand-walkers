#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:57:15 2017

@author: luca
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import time

def disPtLn(m,c,x,y):
    return (-m*x+y-c)/np.sqrt(m**2+1)

def Theta2D(x,y, m,c,side,k=120):
	'''
	Return value of 2D Heaviside Theta with separator being line (m,c)
	'''
	if side == 'upper':
		return 0.5 + 1/np.pi * np.arctan(+k* disPtLn(m,c,x,y))#(-m*x -c +y))
	if side == 'lower':
		return 0.5 + 1/np.pi * np.arctan(-k* disPtLn(m,c,x,y))#(-m*x -c +y))
	else:
		raise Exception('invalid choice of half plane argument for 2d Theta')

def SelectorFn(x,y,vertices):
    '''
    Returns 1 for points inside boundary specified by arbitrary vertices and 0 otherwise
    The points are assumed to define a convex bounded space
    vertices := n by 2 array of coordinates
    '''
    CoM = np.array([np.mean(vertices[:,0]), np.mean(vertices[:,1])])

    flagf = 1
    for nside in range(len(vertices[:,0]) - 1):

        m = (vertices[nside+1,1] - vertices[nside,1])/(vertices[nside+1,0] - vertices[nside,0])
        c = vertices[nside,1] - m * vertices[nside,0]

        if np.sign(-m*CoM[0] - c + CoM[1]) >= 0:
            flagf *= Theta2D(x,y,m,c,'upper')

        else:
            flagf *= Theta2D(x,y,m,c,'lower')

    m = (vertices[0,1] - vertices[-1,1])/(vertices[0,0] - vertices[-1,0])
    c = vertices[0,1] - m * vertices[0,0]

    if np.sign(-m*CoM[0] - c + CoM[1]) >= 0:
        flagf *= Theta2D(x,y,m,c,'upper')

    else:
        flagf *= Theta2D(x,y,m,c,'lower')

    return flagf

def genShaper(x,y,vertices):
    '''
    #rescale x coordinates to fit in 1x1 square
    vertices[:,0] += min(vertices[:,0])
    vertices[:,0] /= max(vertices[:,0])

    #rescale y coordinates to fit in 1x1 square
    vertices[:,1] += min(vertices[:,1])
    vertices[:,1] /= max(vertices[:,1])
    	'''
    shaper = dblquad(lambda a, b: SelectorFn(a,b,vertices)*SelectorFn(x+a,y+b,vertices),
                     0, 1, lambda x: 0, lambda x: 1, epsabs=1e-3)
    return shaper[0]

#vertices = np.array([0.01,0,0,1,0.99,1,1,0.01]) #squre
#vertices= np.array([0,0,0.01,1,1,0.5]) #triangle
#vertices = np.array([0.1,0.3,0.25,0.98,0.9,0.9,0.7,0.4,0.4,0.05])
#vertices = vertices.reshape(int(len(vertices)/2),2)
#resc_vertices = np.copy(vertices)
'''
#rescale x coordinates to fit in 1x1 square
resc_vertices[:,0] += min(vertices[:,0])
resc_vertices[:,0] /= max(resc_vertices[:,0])

#rescale y coordinates to fit in 1x1 square
resc_vertices[:,1] += min(vertices[:,1])
resc_vertices[:,1] /= max(resc_vertices[:,1])
'''

delta = 2 * np.sqrt(2) / 121.
x = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta/2.
y = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta/2.
X, Y = np.meshgrid(x, y)

Z = np.zeros((len(x),len(y)))
start = time.time()
for i,xi in enumerate(x):
    for j,yi in enumerate(y):
        print(str(j + len(y)*i)+' over '+str(len(x)*len(y)))

        if j + len(y)*i == 10:
            stop = time.time()
            print('Predicted runtime: '+str(int(len(x)*len(y)/10.*(stop-start)/60.*5))+' minutes')

        Z[i,j] = round(genShaper(xi,yi,resc_vertices), 3) #gSquare2D(xi+delta/2.,yi+delta/2.,30)
        # Z[i,j] = SelectorFn(xi,yi,resc_vertices)

print('calculations done')

np.save('weird_Z_{:}'.format(len(x)), Z)

# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# plt.figure()
# CS = plt.contour(X, Y, Z, 7,
#                  colors='b',
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
#
# plt.figure()
# plt.contourf(X,Y,Z)
# plt.colorbar()
# print(vertices[:,0],vertices[:,1])
# plt.scatter(vertices[:,0],vertices[:,1])
