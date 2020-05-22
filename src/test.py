# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:53:02 2020

@author: ningyu

"""

import numpy as np
import matplotlib.pyplot as plt
distancetol = 5;

points = [[10,20],[30,40],[50,60],[70,80]]
points = np.asarray(points)
n = len(points[:,0])

plt.figure()
plt.plot(points[:,0],points[:,1],'rs-')

for i in range(n-1,0,-1):
    d = np.linalg.norm(points[i,:]-points[i-1,:]);
    #print('distance = ',d)
    if d > distancetol:
        
        points = np.insert(points,i, \
                           [(points[i-1,0]+points[i,0])/2, \
                            (points[i-1,1]+points[i,1])/2], \
                            axis=0)
        
        #print(i,(points[i-1,0]+points[i,0])/2,(points[i-1,1]+points[i,1])/2)
plt.figure()
plt.plot(points[:,0],points[:,1],'bo-')