# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

N = 100000

# Code for modifying sequence of positions via rotations to make the asymmetry clear
# Valid for 2D case

data = rand.uniform(0,1,size=(2,N))

def Rot_steps(data):
    """ 
    Feed data containing stopping positions in 2D - shape [2 , n-stops]
    """
    rot_steps = np.zeros([2,np.shape(data)[1] - 2])

    for i in range(np.shape(data)[1] - 2):
        
        a = data[:,i]
        b = data[:,i+1]
        c = data[:,i+2]
        
        phi = np.arccos(np.dot(b-a , np.array([0,-1])) / np.linalg.norm(b-a))
        
        if a[0]>b[0]:
            theta = -phi  
        else:
            theta = phi
        
        R = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        c_rot = R.dot(c)
        
        rot_steps[:,i] = c_rot
        
    print('Data created')    
    #for i in range(N-2):
    #    plt.scatter(rot_steps[:,i][0],rot_steps[:,i][1])
    plt.figure()
    plt.hist2d(rot_steps[0,:],rot_steps[1,:],bins=50)
    return 