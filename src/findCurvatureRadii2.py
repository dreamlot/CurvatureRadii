# -*- coding: utf-8 -*-

"""
Renamed from a program from scipy cookbook.
Minor bugs were fixed.
http://www.scipy.org/Cookbook/Least_Squares_Circle
"""

#from numpy import *
import numpy as np

# == METHOD 3 ==
from scipy      import  odr

method_3  = "odr"

def calc_R(c,x,y):
    """ calculate the distance of each 2D points from the center c=(xc, yc) """
    return np.sqrt((x-c[0])**2 + (y-c[1])**2)

def f_3(beta, x):
    """ implicit function of the circle """
    xc, yc, r = beta
    return (x[0]-xc)**2 + (x[1]-yc)**2 -r**2
    #return sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r

def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = data.x.mean(axis=1)
    r0 = np.sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
    #r0 = ((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
    return xc0, yc0, r0

def findCurvatureR(x,y):
    # for implicit function :
    #       data.x contains both coordinates of the points
    #       data.y is the dimensionality of the response
    lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
    lsc_model = odr.Model(f_3, implicit=True, estimate=calc_estimate)
    lsc_odr   = odr.ODR(lsc_data, lsc_model)
    lsc_out   = lsc_odr.run()

    xc_3, yc_3, R_3 = lsc_out.beta
    Ri_3       = calc_R([xc_3, yc_3],x,y)
    residu_3   = sum((Ri_3 - R_3)**2)
    residu2_3  = sum((Ri_3**2-R_3**2)**2)
    #ncalls_3   = f_3.ncalls

    #print ('lsc_out.sum_square = ',lsc_out.sum_square)
    return( [xc_3, yc_3, R_3])


# generate a circle
def getCircle(xyr,n=100):
    print(xyr)
    theta = np.linspace(0,2*np.pi,n);
    x = xyr[0] + xyr[2] * np.cos(theta)
    y = xyr[1] + xyr[2] * np.sin(theta)
    return(x,y)


# test
if __name__ == '__main__':

    x = np.r_[36, 36, 19, 18, 33, 26]
    y = np.r_[14, 10, 28, 31, 18, 26]

    R = 1.0;
    theta = np.linspace(-0.1*np.pi,0.1*np.pi,5);
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    x[2] = x[2] * 0.995
    x[4] = x[4] * 1.003

    [x0,y0,r] = findCurvatureR(x,y)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x,y,'s')
    x,y = getCircle([x0,y0,r])
    plt.plot(x,y)
    plt.axis('square')
