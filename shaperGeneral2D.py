import numpy as np
import mtplotlib.pyplot as plt
from scipy.integrate import dblquad

def 2dTheta(x,y, m,c,side,k=30):
	'''
	Return value of 2D Heaviside Theta with separator being line (m,c)
	'''
	if side = 'upper':
		return 0.5 + 1/np.pi * np.arctan(+k* (-mx -c +y))
	if side = 'lower':
		return 0.5 + 1/np.pi * np.arctan(-k* (-mx -c +y))
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
	for nside in (len(vertices[:,0]) - 1):
	
		m = (vertices[nside+1,1] - vertices[nside,1])/(vertices[nside+1,0] - vertices[nside,0])
		c = vertices[nside,1] - m * vertices[nside,0]
		
		if np.sign(-m*Com[0] - c + CoM[1]) >= 0: 
			flagf *= 2dTheta(x,y,m,c,'upper')
			
		if np.sign(-m*Com[0] - c + CoM[1]) < 0: 
			flagf *= 2dTheta(x,y,m,c,'lower')
			
	return flagf
	
def genShaper(x,y,vertices):
	#rescale x coordinates to fit in 1x1 square
	vertices[:,0] += min(vertices[:,0])
	vertices[:,0] /= max(vertices[:,0])
	
	#rescale y coordinates to fit in 1x1 square
	vertices[:,1] += min(vertices[:,1])
	vertices[:,1] /= max(vertices[:,1])
	
	shaper = dblquad(lambda a, b: SelectorFn(a,b,vertices)*SelectorFn(x+a,y+b,vertices)), 0, 1, lambda x: 0, lambda x: 1)
	return shaper