# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:21:33 2019

@author: ningyu
"""

#from __future__ import print_function
import cv2
import numpy as np
#import imutils
#import ero_dil
#import warnings

def imshow(img,showresult=True,name='result',x=800,y=600):
    if not showresult: 
        return
    
    cv2.namedWindow(str(name),cv2.WINDOW_NORMAL)   
    cv2.imshow(str(name),img);
    cv2.resizeWindow(str(name), x,y)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

# Get the two boundary lines of the single channel
def findSingleChannel(filename,threshold=127,iterations=[2,8,6],showresult=False):
    #filename = "test.jpg"
    print("Reading image : ", filename)
    imReference = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    '''
    # filter image
    imReference = cv2.bilateralFilter(imReference,5,40,40)
    '''
    
    # convert to grayscale
    imgray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    #imgSPLIT = cv2.split(imReference);
    #imgray = imgSPLIT[1];
    imshow(imgray,showresult)
    
    # filter image
    imgray = cv2.bilateralFilter(imgray,5,40,40)
    
    
    # cut image edges
    imReference = imReference[2:-2,2:-2,:]
    
    
    
    
    
    # denoise
    imgray = cv2.fastNlMeansDenoising(imgray,None,10,7,21);
    
    # erode and dilate
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(imgray,kernel,iterations = iterations[0])
    erosion = cv2.erode(dilation,kernel,iterations = iterations[1])
    imshow(erosion,showresult);
    dilation = cv2.dilate(erosion,kernel,iterations = iterations[2])
    imshow(dilation,showresult);
    
    # get the contours
    ret, thresh = cv2.threshold(dilation, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
                                              mode=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imReference, contours, contourIdx = -1, color=(255,0,0), \
                     thickness=4)
    imshow(imReference,showresult);
    
    
    # find the two contours for the two channel boundaries
    # the contours is a list of 3-D numpy arrays
    # x-coordinates of points on the contours are in contours[i][:,0,0]
    # y-coordinates of points on the contours are in contours[i][:,0,1]
    # and we squeeze it.
    
    lengthofcontours = np.zeros(len(contours));
    for ite in range(len(contours)):
        lengthofcontours[ite] = len(contours[ite]);
    #print(lengthofcontours)
    #result = np.array([[]]);
    n_max = np.argmax(lengthofcontours)
    #print(lengthofcontours)
    #print(n_max)
    boundary1 = np.squeeze( contours[n_max] );
    lengthofcontours[n_max] = 0
    
    n_max = np.argmax(lengthofcontours)
    #print(lengthofcontours)
    #print(n_max)
    boundary2 = np.squeeze( contours[ n_max ] );
    
    # remove the figure frame
    x_max = max( max(boundary1[:,0]), max(boundary2[:,0]) )
    x_min = min( min(boundary1[:,0]), min(boundary2[:,0]) )
    y_max = max( max(boundary1[:,1]), max(boundary2[:,1]) )
    y_min = min( min(boundary1[:,1]), min(boundary2[:,1]) )
    
    pos = np.where(boundary1[:,0] == x_max)
    pos = np.append(pos, np.where(boundary1[:,0] == x_min) )
    pos = np.append(pos, np.where(boundary1[:,1] == y_max) )
    pos = np.append(pos, np.where(boundary1[:,1] == y_min) )
    boundary1 = np.delete(boundary1,pos,0);
    
    
    pos = np.where(boundary2[:,0] == x_max)
    pos = np.append(pos, np.where(boundary2[:,0] == x_min) )
    pos = np.append(pos, np.where(boundary2[:,1] == y_max) )
    pos = np.append(pos, np.where(boundary2[:,1] == y_min) )
    boundary2 = np.delete(boundary2,pos,0);
    
    tmp = np.copy(boundary1[:,0])
    boundary1[:,0] = boundary1[:,1]
    boundary1[:,1] = np.copy(tmp)
    tmp = np.copy(boundary2[:,0])
    boundary2[:,0] = boundary2[:,1]
    boundary2[:,1] = np.copy(tmp)
    
    y1 = np.average(boundary1[:,1])
    y2 = np.average(boundary2[:,1])
    
    if y1 < y2:
        result = [boundary1, boundary2];
    else:
        result = [boundary2, boundary1];
    
    return( result )
    

def createContourImage(contour,shape=None):
    if shape is None:
        x = np.ceil(max(contour[0][:,0],contour[1][:,0])) \
            - np.floor(min(contour[0][:,0],contour[1][:,0]))
        y = np.ceil(max(contour[0][:,1],contour[1][:,1])) \
            - np.floor(min(contour[0][:,1],contour[1][:,1]))
        shape = (x,y)
        
    contourimg = np.zeros(shape)
    contourimg[contour[0][:,0].astype(int),contour[0][:,1].astype(int)] = 255;
    contourimg[contour[1][:,0].astype(int),contour[1][:,1].astype(int)] = 255;
    
    contourimg = np.uint8(contourimg)
    #contourimg.astype(int);
    #contourimg.shape = (contourimg.shape[0],contourimg.shape[1],1)
    
    return( contourimg )
    
def alignImage(contourref,contour,img):
    
    # Find size of image1
    sz = contourref.shape


    '''
    First, do translational fix.
    '''
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    #warp_mode = cv2.MOTION_HOMOGRAPHY
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 1000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-6;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                number_of_iterations,  termination_eps)
    
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (contourref,contour,warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (contour, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        result = cv2.warpPerspective (img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(contour, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        result = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
     
    N_ITERATION_OUTER = 6;
    N_ITERATION_START = 500;
    N_ITERATION_END = 10000;
    EPS_START = 1e-5;
    EPS_END = 1e-8;
    for lp1 in range(N_ITERATION_OUTER):
        '''
        Second, do rotational fix.
        '''
        # Define the motion model
        warp_mode = cv2.MOTION_EUCLIDEAN
        #warp_mode = cv2.MOTION_HOMOGRAPHY
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        '''
        if lp1 == N_ITERATION_OUTER - 1:
            number_of_iterations = N_ITERATION_END;
            termination_eps = EPS_END;
        else:
            number_of_iterations = N_ITERATION_START;
            termination_eps = EPS_START;
        '''
        number_of_iterations = int( ( lp1*N_ITERATION_END + (N_ITERATION_OUTER-lp1)*N_ITERATION_START ) / N_ITERATION_OUTER )
        termination_eps =  ( lp1*EPS_END + (N_ITERATION_OUTER-lp1)*EPS_START ) / N_ITERATION_OUTER
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                    number_of_iterations,  termination_eps)
        
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (contourref,im2_aligned,warp_matrix, warp_mode, criteria)
         
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            result = cv2.warpPerspective (result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            result = cv2.warpAffine(result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
         
        
        '''
        Third, do translational fix, again
        '''
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
        #warp_mode = cv2.MOTION_HOMOGRAPHY
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        '''
        if lp1 == N_ITERATION_OUTER - 1:
            number_of_iterations = N_ITERATION_END;
            termination_eps = EPS_END;
        else:
            number_of_iterations = N_ITERATION_START;
            termination_eps = EPS_START;
        '''
        number_of_iterations = int( ( lp1*N_ITERATION_END + (N_ITERATION_OUTER-lp1)*N_ITERATION_START ) / N_ITERATION_OUTER )
        termination_eps =  ( lp1*EPS_END + (N_ITERATION_OUTER-lp1)*EPS_START ) / N_ITERATION_OUTER
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \
                    number_of_iterations,  termination_eps)
        
        
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (contourref,im2_aligned,warp_matrix, warp_mode, criteria)
         
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im2_aligned = cv2.warpPerspective (im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            result = cv2.warpPerspective (result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2_aligned, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            result = cv2.warpAffine(result, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
         
        
    # Show final results
    #imshow(im2_aligned[0],name="Aligned Image 2",showresult=True )

    return( im2_aligned, result )
    
    
    
    
    
    
if __name__ == '__main__':
    
    workpath = '.';
    filename = 'reference.jpg';
    thre = 127;
    iteration = [0,4,4]
    try:
        fp = open('control.txt','r')
        workpath = fp.readline();
        for i in range(2):
            if workpath[-1] == '\n':
                workpath = workpath[0:-1]
            if workpath[-1] == '\r':
                workpath = workpath[0:-1]
        filename = fp.readline();
        for i in range(2):
            if filename[-1] == '\n':
                filename = filename[0:-1]
            if filename[-1] == '\r':
                filename = filename[0:-1]
        thre = int(fp.readline());
        fp.close();
            
    except:
        pass
    
        
    ''' 
    workpath = 'C:/Users/ningyu/Dropbox/PhD_work/ferrofluid_experiment/20191018'
    #workpath = 'C:/Users/ningyu/Dropbox/PhD_work/ferrofluid_experiment/20191003postprocess/View 1';
    '''
    
    #filename = workpath+"/reference.jpg"
    filename = workpath +"/" + filename
    cha1 = findSingleChannel(filename,iterations=iteration);
    #cha1 = findSingleChannel(filename,threshold=thre,iterations=[1,2,1]);
    
    import matplotlib.pyplot as plt
    '''
    plt.figure();
    plt.plot(cha1[0][:,0],cha1[0][:,1],'o-')
    plt.plot(cha1[1][:,0],cha1[1][:,1],'s-')
    '''
    
    # cut the reference frame and every other frames
    MARGINX = 50;
    MARGINY = 30;
    
    
    # make reference contour frame
    # notice that contour and imshow are transposed from each other
    imReference = cv2.imread(filename, cv2.IMREAD_COLOR)
    SHAPE = imReference[:,:,0].shape;
    '''
    contourref = np.zeros(imReference[:,:,0].shape)
    contourref[cha1[0][:,0].astype(int),cha1[0][:,1].astype(int)] = 255;
    contourref[cha1[1][:,0].astype(int),cha1[1][:,1].astype(int)] = 255;
    plt.figure()
    #plt.imshow(255-contourref)
    plt.contourf(contourref)
    '''    
    contourref = createContourImage(contour=cha1,shape=SHAPE)
    cv2.imwrite(workpath+'/contourref.tif',contourref)
    '''
    plt.figure()
    plt.contourf(contourref)
    '''
    
    # get the list of all figures in the working directory
    import os
    pathorigin = workpath+'/origin';
    files = os.listdir(pathorigin)
    pathtarget = workpath+'/target';
    try:
        os.mkdir(pathtarget);
    except FileExistsError:
        pass
        #warnings.warn('Subfolder ' + pathtarget + ' already exists!')
    #img = cv2.imread(pathorigin+'/'+files[0])
    #imshow(img)
    
    
    # make contour 
    for ite in enumerate(files):
        filename = pathorigin+'/'+ite[1];
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        #cha2 = findSingleChannel(filename,iterations=[1,2,1]);
        cha2 = findSingleChannel(filename,threshold=thre,iterations=iteration);
        
        
        contour = createContourImage(contour=cha2,shape=SHAPE)
        #imshow(contourref[0])
        savename = pathtarget+'/con'+ite[1]+".tif";
        cv2.imwrite(savename,contour)
        
        imgaligned, result = alignImage(contourref,contour,img)
        
        #imshow(imgaligned)
        #savename = pathtarget+'/'+ite[1];
        #cv2.imwrite(savename,imgaligned)
        savename = pathtarget+'/res'+ite[1]+".tif";
        cv2.imwrite(savename,result)
        
        #break
        
        '''
        tmp = cha2[0][:,0]
        cha2[0][:,0] = cha2[0][:,1]
        cha2[0][:,1] = tmp
        tmp = cha2[1][:,0]
        cha2[1][:,0] = cha2[1][:,1]
        cha2[1][:,1] = tmp
        
        cha2[0] = cha2[0].reshape((-1,1,2)).astype(np.int32)
        cha2[1] = cha2[1].reshape((-1,1,2)).astype(np.int32)
        
        cv2.drawContours(img, cha2, contourIdx = -1, color=(255,0,0), thickness=4)
        cv2.imwrite(pathtarget+'/con'+ite[1],img)
        '''