# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 09:41:02 2017

@author: men14
"""
import numpy as np
from data_generation import weird_bounds, DelaunayArray, in_bounds, Delaunay
from scipy.interpolate import griddata
import cv2

def rotation(x,y,angle):
    """
    rotates x,y position for angle theta
    """

    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    return x_rot, y_rot


def radial_interp(data, xcentre, ycentre, num_radii, num_points_per_radius, dtype='float'):
    """
    Does radially interolation to average radially to get radial shape on
    given grid shape
    centre is the centre from which the radial process starts

    """
    if dtype=='float':
        data_copy = np.zeros(data.shape, dtype=np.int32)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row,col] > 1e-10:
                    data_copy[row,col] = 1
    elif dtype == 'int32':
        data_copy = data.copy()
    else:
        raise Exception('Error: not correct integer type of input data')

    # generate mask
    filled_array = np.zeros((data_copy.shape[0]+2, data_copy.shape[1]+2), np.uint8)
    cv2.floodFill(data_copy, filled_array, (0,0), newVal=255)

    mask = np.zeros((xcentre.size,ycentre.size), dtype=bool)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if data_copy[row,col] != 255:
                mask[row,col] = True
            else:
                mask[row,col] = False

    data_array = data[mask]

    pos = np.zeros((data_array.size,2)) - 9

    # create x and y arrays
    pos_array = np.zeros((data.shape[0], data.shape[1], 2)) - 9
    for col in range(data.shape[1]):
        pos_array[:,col,0] = xcentre
    for row in range(data.shape[0]):
        pos_array[row,:,1] = ycentre

    pos[:,0] = pos_array[mask,0] # x positions
    pos[:,1] = pos_array[mask,1] # y positions

    max_x = np.max(pos[:,0])
    max_y = np.max(pos[:,1])
    max_rad = np.sqrt(max_x**2 + max_y**2)

    # interpolate on grid encompassing these points
    points = []
    radii = np.zeros(num_radii + 1)
    index = 0
    for r in np.arange(0, max_rad + max_rad / float(num_radii), max_rad / float(num_radii)):
        x = r
        y = 0.
        radii[index] = r
        for angle in np.arange(0, 2*np.pi, 2*np.pi / num_points_per_radius):
            #print('r', r, x, y)
            # rotate around z axis
            x,y = rotation(x,y,angle)
            points.append([x,y])
        index += 1

    interp_points = griddata(pos, data_array, points, fill_value=-9.0)

    # average number of points at same radius
    avg = np.zeros(num_radii + 1) - 9
    start = 0
    for i in range(num_radii + 1):
        values = interp_points[start*num_points_per_radius:(start+1)*num_points_per_radius]
        index = np.where(values != - 9.0)[0]
        if index.size != 0:
            avg[i] = np.average(values[index])
        else:
            avg[i] = 0.
        start += 1

    # normalising averages weighted by area
    total = np.sum(avg) * max_rad / float(num_radii)
    avg /= total

    return avg, radii


