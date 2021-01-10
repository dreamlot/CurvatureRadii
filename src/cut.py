# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:52:35 2019

Cut the figure.

@author: Ningyu Wang
"""


import cv2

# Cut the given OpenCV image item and return the result. im is a 2D or 3D np array.
# By default, x and y are ranges in percentage, which means they are whithin
# the range of [0.0,1.0]. If FlagFlagPercent is set to False, the cutting would be
# based on pixels.
def cut(im, x=[0.0,1.0], y=[0.0,1.0], FlagPercent=True):
    if FlagPercent:
        if im.ndim == 2:
            nx,ny = im.shape;
            im = im[int(x[0]*nx):int(x[1]*nx), int(y[0]*ny):int(y[1]*ny)];
        elif im.ndim == 3:
            nx,ny,nz = im.shape;
            im = im[int(x[0]*nx):int(x[1]*nx), int(y[0]*ny):int(y[1]*ny),:];
    else:
        if im.ndim == 2:
            im = im[x[0]:x[1], y[0]:y[1]];
        elif im.nim == 3:
            im = im[x[0]:x[1], y[0]:y[1],:];
    return(im)

# Cut all images in a directory and store the results in a new path. im is a
# 2D or 3D np array.
# By default, x and y are ranges in percentage, which means they are whithin
# the range of [0.0,1.0]. If FlagFlagPercent is set to False, the cutting would be
# based on pixels.
def cutall(x=[0.0,1.0],y=[0.0,1.0],sourcepath='.',targetpath='.'):
    # get the list of all figures in the working directory

    import os
    from os.path import isfile, join
    files = [f for f in os.listdir(sourcepath) if isfile(join(sourcepath, f))]

    # create the targe directory
    try:
        os.mkdir(targetpath);
    except FileExistsError:
        pass
    # cut the images and save to the target directory
    for ite in enumerate(files):
        # skip Windows image preview files
        if ite[1][-3:] == 'ini': continue

        filename = sourcepath+'/'+ite[1];
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        # cut the image
        im = cut(im,x=x,y=y)
        savename = targetpath+'/'+ite[1];
        cv2.imwrite(savename,im)


if __name__ == '__main__':

    workpath = '.';
    filename = 'reference.jpg';
    try:
        fp = open('control.txt','r')
        workpath = fp.readline();
        filename = fp.readline();
        for i in range(2):
            if workpath[-1] == '\n':
                workpath = workpath[0:-1]
            if workpath[-1] == '\r':
                workpath = workpath[0:-1]
        fp.close();
    except:
        pass

    # get the list of all figures in the working directory
    pathorigin = workpath+'/origin';
    try:
        os.mkdir(pathorigin);
    except FileExistsError:
        pass
    files = os.listdir(workpath)

    # cut the images and save to the target directory
    for ite in enumerate(files):
        if ite[1][-3:] != 'jpg':
            #print(ite[1][-3:])
            continue
        filename = workpath+'/'+ite[1];
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        # cut the image
        im = cut(im,[0.35,0.7])
        savename = pathorigin+'/'+ite[1]+'.tiff';
        cv2.imwrite(savename,im)
