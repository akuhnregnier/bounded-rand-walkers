# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:55:33 2017

@author: men14
"""
from scipy.stats import norm
import numpy as np

def gaussian(centre, scale, point, dimensions):
    """
    This function calculates the probability at point 'point' given a (in 2d
    roationally symmentric) Gaussian.
    
    Args:
        centre: centre of the Gaussian
        scale: used to scale the Gussian
        point: the point at which the probability should be found
        dimensions: dimensions of the space i.e. 1 or 2D
        
    Result: 
        prob: probability at 'point'
    
    """
    
    if dimensions == 1:
        prob = norm.pdf(point, loc=centre, scale=scale)
        
        return prob
    
    elif dimensions == 2:
        # symmentric in radius
        prob = norm.pdf(np.sqrt(point[0]**2 + point[1]**2), loc=centre,
                        scale=scale)
        
        return prob