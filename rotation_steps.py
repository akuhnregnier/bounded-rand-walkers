# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

N = 1000000

# Code for modifying sequence of positions via rotations to make the asymmetry clear
# Valid for 2D case

data = rand.uniform(0,1,size=(2,N))
print('Data created') 

def Rot_steps(data):
    """ 
    Feed data containing stopping positions in 2D - shape [2 , n-stops]
    """
    rot_steps = np.zeros([2,np.shape(data)[1] - 2])

    for i in range(np.shape(data)[1] - 2):
        
        a = data[:,i]
        b = data[:,i+1]
        c = data[:,i+2]
        
        phi = np.arccos(np.dot(a-b , np.array([0,-1])) / np.linalg.norm(a-b))
        
        if a[0]>b[0]:
            theta = -phi  
        else:
            theta = phi
        
        R = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        #print(R.dot())
        c_rot = R.dot(c-b)
        
        rot_steps[:,i] = c_rot
           
    plt.figure()
    plt.hist2d(rot_steps[0,:],rot_steps[1,:],bins=50)
    plt.plot(np.arange(-2,2,0.01), np.array([0 for a in np.arange(-2,2,0.01)]))
    plt.title('Observed step-size with fixed incoming direction')
    return 

Rot_steps(data)