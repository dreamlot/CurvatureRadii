# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:58:52 2020

@author: Ningyu Wang
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

gradient = np.linspace(0, 1, 256)
#gradient = np.vstack((gradient, gradient))

#fig = plt.figure()

#plt.imshow(gradient, aspect='auto', cmap='summer')


fig = plt.figure()
z = np.zeros(gradient.shape)
#img = np.array([z,z,gradient])
img = np.zeros((len(gradient),20,3))
for ite in range(20):
    img[:,ite,0] = gradient
h = plt.imshow(img)

#cbar = np.array()
plt.ylim(0,256)
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.yaxis.tick_right()
plt.yticks(ticks=[0,128,256],labels=[17,26,34])
plt.title(r'$\mu m$')
