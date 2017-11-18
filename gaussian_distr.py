# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:55:33 2017

@author: men14
"""
from scipy.stats import norm
import numpy as np

def gaussian1d(x, centre, scale):
    """
    This function calculates the probability at point x given Gaussian.
    
    Args:
        centre: centre of the Gaussian
        scale: used to scale the Gussian
        x: the x position at which the probability should be found
        
    Result: 
        prob: probability at x
    
    """
    
    prob = norm.pdf(x, loc=centre, scale=scale)
        
    return prob




def gaussian2d(x, y, centre, scale):
    """
    This function calculates the probability at point x,y given a roationally
    symmentric Gaussian.
    
    Args:
        centre: list of x,y position of the centre of the Gaussian
        scale: used to scale the Gussian
        x: xposition at which the probability should be found
        y: yposition at which the probability should be found
        
    Result: 
        prob: probability at 'x,y'
    
    """
    
    # symmentric in radius
    prob = norm.pdf(np.sqrt((x - centre[0])**2 + (y - centre[1])**2),
                    scale=scale)
    
    return prob