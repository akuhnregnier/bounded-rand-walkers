# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:22:19 2017

@author: men14
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

def Binning1D(binsize, data, normalising):
    """
    This function bins the data in constant binsizes. 
    
    Args:
        data: 1d array of data that should be binned
        binsize: constant, which is the size of each bin
        normalising: Boolean, if true, the histogram is normalised
    
    returns:
        histrogram: list of list with [centre of bin, frequency]
        
    """

    if type(data) == list:
        data = np.asarray(data)
    if len(data) == 0:
        return []


    n_elements = float(data.size)
    histogram = []
       
    # shift everything by number binsizes upwards and then shift it down
    min_value = np.min(data)
    bin_num = np.ceil(np.abs(min_value / float(binsize)))
    
    data = data + bin_num * binsize
   
    
    # Bin the positive values
    flag = True
    counter = 1.
    while flag == True:
        indeces = np.where((data - counter * binsize) < 0)[0]
        
        # Normalise it if desired
        if normalising:
            freq = indeces.size / n_elements
        else:
            freq = indeces.size 
        
        data = np.delete(data, indeces)
        
        if freq > 0:
            # binsize / 2 so get centre bin
            histogram.append([binsize * (counter - 0.5 - bin_num), freq]) 
        
        if not data.size:
            flag = False
        else:
            counter += 1

    return histogram



def Binning2D(binsize, data, normalising):
    """
    This function bins the data in constant binsizes in 2 dimensions
    
    Args:
        data: 2d array of data that should be binned
        binsize: list of constant x and y binsize ie [x_binsize, y_binsize]
        normalising: Boolean, if true, the histogram is normalised
    
    returns:
        histrogram: list of list with [ [x,y centre of bin], frequency]
        
    """

    if type(data) == list:
        data = np.asarray(data)
    if len(data.shape) == 0:
        raise Exception('Passed empty data set to 2D binning function')
        
    min_value = np.min(data[:,0])
    bin_num = np.ceil(np.abs(min_value / float(binsize[0])))
    
    copy_data = copy.deepcopy(data)
    
    copy_data[:,0] = data[:,0] + bin_num * binsize[0]

    flag = True
    n_elements = data.size / 2.
    histogram2D = []
    counter = 1.
    
    while flag == True:
        indeces = np.where((copy_data[:,0] - counter * binsize[0]) < 0)[0]
        
        ybin = Binning1D(binsize[1], copy_data[indeces,1], False)
        
              
        copy_data = np.delete(copy_data, indeces, axis = 0)
        
        if ybin > 0:
            for l in range(len(ybin)):
                if normalising:
                    histogram2D.append([[binsize[0] * (counter - 0.5 - bin_num),
                                         ybin[l][0]], ybin[l][1] / n_elements])
                else:
                    histogram2D.append([[binsize[0] * (counter - 0.5 - bin_num),
                                         ybin[l][0]], ybin[l][1]])
        
        if not copy_data.size:
            flag = False
        else:
            counter += 1
    
    return histogram2D
            
        

def Estimate_Fi(length_list, binsize):
    """
    This function uses all lengths lists to estimate the fi(y) distribution.
    
    Args:
        length_list: list of all step sizes
    
    return:
        histogram of fi(y)
        
    """
    
    histogram = Binning1D(binsize, length_list, True) 
       
    return histogram


def Estimate_Gx(binsize, length_list):
    """
    This function estimates Gx by first taking the cumulative sum of the length
    list and then binning the result.
    
    Args:
        length_list: list or array of all step sizes 
        binsize: size of the bins in which the data should be put
    
   Note: both input arguments can either be 1 or 2D 
   
    """

    # Calculate cumulative list
    cum_array = np.cumsum(length_list, axis = 0)
    print '3'
    
    if len(cum_array.shape) == 1:
        binned = Binning1D(binsize, cum_array, True)
        
        return binned
    
    elif len(cum_array.shape) == 2:
        binned = Binning2D(binsize, cum_array, True)
        xpos, ypos, freq_array = ConvertArray(binned, binsize)
        
        return xpos, ypos, freq_array        
    
    else:
        raise Exception('The input array is neither 1 or 2D which this process'
                        ' cannot handel')
    return binned

def ConvertArray(histogram2D, binsize):
    """
    This function converts an input list of list, of the form [[[x,y], freq],
    [[x2,y2], freq2]] into the one list of all xpositions, one list of all
    ypositions and an 2D array of shape (y,x) of the freq values. This
    fuction would be usually be used to convert the output of Binning2D into
    an usable array format.
    
    Args:
        histogram2D: a list of list of the format described above
        binsize: a 1D array or list of the binsizes in the x,y direction
        
    Results:
        all_xpos: list of all xpositions starting from the minimum x position,
                  ending at the max xposition in integer binsize steps
        all_ypos: same as all_xpos just with y values and y binsizes
        freq_array: 2D array of shape (y,x) with their frequencies as entries
    
    """  
    
    xpos = [histogram2D[i][0][0] for i in range(len(histogram2D))]
    ypos = [histogram2D[i][0][1] for i in range(len(histogram2D))]
    
    min_x = np.min(xpos)
    max_x = np.max(xpos)
    
    min_y = np.min(ypos)
    max_y = np.max(ypos)
    
    freq_array = np.zeros((int(np.round(np.floor((max_y - min_y) / float(binsize[1])))) + 2,
                           int(np.round(np.floor((max_x - min_x) / float(binsize[0])))) + 2))
    
    print freq_array.shape
        
    for i in range(len(histogram2D)):
        xindex = (histogram2D[i][0][0] - min_x) / float(binsize[0])
        yindex = (histogram2D[i][0][1] - min_y) / float(binsize[1])
        
       
        freq_array[int(np.round(yindex)), int(np.round(xindex))] = histogram2D[i][1]
    
    all_xpos = []
    all_ypos = []
    
    for i in range(freq_array.shape[1]):
        all_xpos.append(min_x + i * binsize[0])
    for j in range(freq_array.shape[0]):
        all_ypos.append(min_y + j * binsize[1])
    
    return all_xpos, all_ypos, freq_array
    
    

'''
Testing stuff out, this can be removed
'''  

#data = [-1.1,-1.2,-1.3,4,5,6,6.7,8,8,10]
#print Binning1D(1, data, True)

#data2 = [[-1.1, -1.3],[1.2, 1.7], [1.3, 2.4] ,[-3.9,0], [5,6] , [6.7,-8.2] , [8,8] ,[10,2.4]]
#print Binning2D([1,1], np.asarray(data2), False)
#data2 = [[-1.1, -1.1],[0.1, 0.1], [1.1, 1.1] ,[2.1,2.1], [3.1,3.1]]


data = np.random.normal(size=(20000,2))
print np.min(data, axis = 0)
print np.max(data, axis = 0)


#new_data = Estimate_Gx([0.1,0.1], np.asarray(data))
binsize = [0.15,0.15]
binned = Binning2D(binsize, data, False)

xpos, ypos, freq_array = ConvertArray(binned, binsize)
#print xpos
print '1'
#plt.figure(1)
#plt.hexbin(*data.T)

#xpos = [new_data[i][0][0] for i in range(len(new_data))]
#ypos = [new_data[i][0][1] for i in range(len(new_data))]
#value = [new_data[i][1] for i in range(len(new_data))]
plt.figure(2)
plt.pcolormesh(xpos, ypos, freq_array)
print '2'
xpos2, ypos2, freq_array2 = Estimate_Gx(binsize, data)
plt.figure(3)
plt.pcolormesh(xpos2, ypos2, freq_array2)
#plt.show()



