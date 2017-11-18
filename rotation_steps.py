# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import scipy as sp

N = 1000

# Code for modifying sequence of positions via rotations to make the asymmetry clear
# Valid for 2D case

#pos_data2D = rand.uniform(0,1,size=(2,N))
#print('Data2D created') 

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


def Corr_spatial_1D(data,binnumber):
    
    binwidth = 1./binnumber
    print(binwidth)
    correlations = []

    for i in range(binnumber):
        
        prev = []
        post = []
        prod = []
    
        for j in range(np.shape(data)[1]-2):
            if i*binwidth < data[0,j+1]< (i+1)*binwidth:
                
                prev.append(data[0,j+1] -data[0,j])
                post.append(data[0,j+2] -data[0,j+1])
                prod.append((data[0,j+1] -data[0,j])*(data[0,j+2] -data[0,j+1]))
                
        corr = np.mean(prod) - np.mean(post)*np.mean(prev)
        correlations.append(corr)
    
    print(correlations)    
    plt.bar([binwidth*i for i in range(binnumber)], correlations,width=binwidth, alpha=0.4)
    
        
def Pdf_Transform(step,f,geometry):
    
    '''
    For a given intrinsic step size pdf f gives probability p(stepsize) of transformed pdf
    Geometry is a str, choices atm '1Dseg' (takes float steps) & '1circle' (takes 2 element arrays)
    '''
    
    if geometry == '1Dseg':
        
        if type(step) != float:
            raise TypeError('for 1D pdf use float for step')
            
        return (1-np.abs(step)) * f(step) * 0.5*(np.sign(1-np.abs(step)) + 1)
    
    if geometry == '1circle':
        
        if type(step) != np.ndarray:
            raise TypeError('for 2D pdf use 1d, 2 entry array, for step')
        
        l = np.linalg.norm(step)
        return f(step) * (2 * np.arccos(l/2) - 0.5 * np.sqrt((4-l**2)*l**2))
    
def g1D(x,f):
    num = sp.integrate.quad(f,-x,1-x)
    den = sp.integrate.quad(lambda z: sp.integrate.quad(lambda y: f(y), -z,1-z), 0,1)
    return num/den
                
    

#Rot_steps(pos_data)