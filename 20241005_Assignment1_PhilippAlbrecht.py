#!/usr/bin/env python
# coding: utf-8


#import modules 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


#prepare data 
#add each line of spectrum.txt to a list. exclude information in the first couple of lines  
def create_list():  
    filename = 'spectrum.txt'
    data = [] 
    with open(filename, 'r') as file: 
        x=0
        for line in file: 
            x+=1
            if x>27: 
                wavelength, flux = line.strip().split(',')
                data.append((float(wavelength), float(flux)))
    return data

#split the data list into a list with the wavelength values and a list with the flux values  
mydata = create_list()
wlist=[] 
flist=[] 

def split_list(): 
    for item, (w, f) in enumerate(mydata): 
        wlist.append(w)
        flist.append(f)

split_list()


# turn wavelength list and flux list into arrays. Note: There is probably a simpler way of doing this - optimization potential 
w_array = np.array(wlist) #wavelength (x-axis)
f_array = np.array(flist) #flux (y-axis) 


#extract values within a defined range. The following two functions were copied from the Brightspace document 'fitting-to-spectra-2102'
from numpy import inf as INF


#Boolean mask with value True for x in [xmin, xmax)
def in_interval(x, xmin=-INF, xmax=INF):     
    _x = np.asarray(x) 
    return np.logical_and(xmin <= _x, _x < xmax) 

def filter_in_interval(x, y, xmin, xmax):
    _mask = in_interval(x, xmin, xmax)     #Selects only elements of x and y where xmin <= x < xmax.
    return [np.asarray(x)[_mask] for x in (x, y)]   


#create arrays that only include the region of interest 
ROI = (6680, 6695)     #ROI estimated based on plot of measurement data. area that includes the measurement peak 
_w_array, _f_array = filter_in_interval(w_array, f_array, *ROI) 


#define polynomial, gaussian and combined function for the fitting curve 
TWO_PI = np.pi * 2

def polynomial_function (x, b, c, d, e): 
    return b * x**3 + c * x**2 + d * x + e     #polynomial function of the third degree. resembles and s-curve 

def gaussian_function (x, mu, sig, a): 
    return a * np.exp(-0.5 * (x-mu)**2 / sig**2) / np.sqrt(TWO_PI * sig**2)

def combined_function (x, b, c, d, e, mu, sig, a): 
    return polynomial_function(x, b, c, d, e) + gaussian_function(x, mu, sig, a)


#estimate parameters for gaussian curve 
mu_guess = np.max(_w_array)     #the highest point of the ROI. This should be at the mean value as a gaussian curve is symmetrical 
sig_guess = np.std(_w_array)     #standard defiation of the curve 
a_guess = 80     #amplitude of the gaussian is estimated based on the plot                       #works with this: --> area = np.trapz(_f_array, _w_array)     #area below the curve in the ROI #function provided by Microsoft Copilot  


#return optimized parameters combined function (polynomial + gaussian)  
initial_guess = [0.1, -0.5, 1, 1, mu_guess, sig_guess, a_guess]             #initial guess for polynomial function taken from online literature                        
popt, pcov = curve_fit(combined_function, w_array, f_array, p0=initial_guess) #The curve_fit function uses non-linear least squares to fit the model function to the data (source: Microsoft Copilot) 

b, c, d, e = popt[:4]     #optimized parameters for polynomial function for continuum 
mu, sig, a = popt[4:]     #optimized parameters for polynomial function for gaussian 


#plot the measurement data 
plt.scatter(w_array, f_array, s=1, label='Measurement data') 
plt.xlabel('Wavelength in Angstrom') 
plt.ylabel('Flux in ADU') 
plt.title('Measurement data') 
plt.legend()
plt.show()


#plot the polynomial curve
fig, ax = plt.subplots()
ax.scatter(w_array, f_array, s=1, label='Measurement data')
ax.plot(w_array, polynomial_function(w_array, b, c, d, e), color='red', label='Continuum function')
plt.xlabel('Wavelength in Angstrom') 
plt.ylabel('Flux in ADU') 
plt.title('Measurement data with continuum function') 
plt.legend()
parameters_text1 = f'''Parameters of the continuum function: \n
     b = {popt[0]:.5f} \n
     c = {popt[1]:.2f} \n
     d = {popt[2]:.2f} \n
     e = {popt[3]:.2f} \n'''
ax.text(1.1, 0.95, parameters_text1, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)) #This line was provided by Microsoft Copilot, parameters were adjusted 
plt.show()


#plot the combined function and a separate box with the parameters 
fig, ax = plt.subplots()
ax.scatter(w_array, f_array, s=1, label='Measurement data') 
ax.plot(w_array, combined_function(w_array, *popt), color='red', label='Fitted function')     #prints fitted function with optimized parameters
plt.xlabel('Wavelength in Angstrom') 
plt.ylabel('Flux in ADU') 
plt.title('Measurement data with fitted function') 
plt.legend()
parameters_text2 = f'''Parameters of the Gaussian function: \n
     Mean value: {popt[4]:.2f} \n
     Standard deviation: {popt[5]:.2f} \n
     Amplitude: {popt[6]:.2f}\n'''
ax.text(1.1, 0.95, parameters_text1 + "\n" + parameters_text2, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)) #This line was provided by Microsoft Copilot, parameters were adjusted 
plt.show()


print(parameters_text1)
print(parameters_text2)
