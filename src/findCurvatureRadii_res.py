# -*- coding: utf-8 -*-
"""
This program is to find the curvature radii along the oil blob perimeter.
It studies only the oil blob on the left hand side.
The figure will be cut in half and the right hand side half image would 
be abandoned.


Created on Thu Feb 27 21:27:19 2020

@author: ningyu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from findCurvatureRadii2 import findCurvatureR

# show image
def imshow(img,showresult=True,name='result',x=800,y=600):
    if not showresult: 
        return

    cv2.namedWindow(str(name),cv2.WINDOW_NORMAL); 
    cv2.imshow(str(name),img);
    cv2.resizeWindow(str(name), x,y);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    
# Get the oil blob perimeter
def findOilBlob(filename,threshold=127,iterations=[2,8,6],showresult=False):
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
    
    # cut half of the image 
    # use only the LHS
    imgray = imgray[:,0:int(imgray.shape[1]/2)];
    
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
    #ret, thresh = cv2.threshold(dilation, 127, 255, 0)
    ret, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_OTSU)
    tmpim, contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
                                              mode=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgray, contours, contourIdx = -1, color=(255,0,0), \
                     thickness=4)
    imshow(imgray,showresult);
    
    # First, the real interfaces have a lot of points. 
    # Abandon the contours with only several points.
    # Keep only the longest contours.
    contours = sorted(contours,key=len);
    contours = contours[0:3];
    
    # Then, there are three longest contours found: 
    # 1. the upper glass, average y value largest
    # 2. the lower glass, average y value smallest
    # 3. the oil blob, average y values in the middle
    #
    # keep the oil blob
    def yavg(x):
        y = np.zeros(3);
        for i in range(3):
            y[i] = np.mean(x[:,:,1])
        #print(y)
        return(np.mean(y))
    contours = sorted(contours,key=yavg)
    '''
    # test if the contours have been sorted based on the average y value
    y = np.zeros(3);
    for i in range(len(contours)):
        y[i] = np.mean(contours[i][:,:,1])
    print(y)
    '''
    
    return (imgray,contours)

# Function of a circle
def funcCircle(para,x,y):
    return( (x-para[0])**2 + (y-para[1])**2 - para[2]**2)
    #return( np.sqrt((x-para[0])**2 + (y-para[1])**2) - para[2])


# from a set of x y values, fit a circle
def findCircle(points):
    try:
        x = points[:,0];
        y = points[:,1];
        x_guess = (max(x)+min(x))/2;
        y_guess = (max(y)+min(y))/2;
        R_guess = ( max(y)-min(y) ) / 2;
        guess = np.array([x_guess,y_guess,R_guess]);
        #print(guess)
        lower_bounds = [min(x),min(y),0]
        upper_bounds = [max(x),max(y),max(max(x),max(y))/2]
        bounds = (lower_bounds,upper_bounds)
    except:
        raise Exception(points.shape)
    return(optimize.least_squares(fun=funcCircle,x0=guess, \
                                  xtol = 1e-12,bounds=bounds,args=(x,y)).x)

# find Curvature
def findCurvatureRadius(points,window=5):
    # number of points
    N = points.shape[0];
    
    # window length used to fit the circle
    n = window;
    
    print(N)
    # fitted circle
    result = np.zeros([N,3]);
    
    for i in range(N):
        if i < N-n:
            indx = np.arange(i,(i+n))
        else:
            indx = np.concatenate((np.arange(i,N),np.arange(0,i+n-N)))
        #result[i,:] = findCircle(points[indx,:]);
        x = points[indx,0]
        y = points[indx,1]
        result[i,:] = findCurvatureR(x,y)
    return( result )
    


# Function of a eclipse
def funcEclipse(para,x,y):
    return( (((x-para[0])*np.cos(para[4])+(y-para[1])*np.sin(para[4]))/para[2])**2 \
             + ((-(x-para[0])*np.sin(para[4])+(y-para[1])*np.cos(para[4]))/para[3])**2 \
             - 1 )
    #return( np.sqrt((x-para[0])**2 + (y-para[1])**2) - para[2])

'''
# from a set of x y values, fit a eclipse
input:
    points: n x 2, np.array
output:
        : list:
            x0:     x coordinate of eclipse
            y0:     y coordinate of eclipse
            a:      semi-major axis
            b:      semi-minor axis
            theta:  tilt angle
'''
def findEclipse(points):
    try:
        x = points[:,0];
        y = points[:,1];
        x_guess = (max(x)+min(x))/2;
        y_guess = (max(y)+min(y))/2;
        R_guess = ( max(y)-min(y) ) / 2;
        guess = np.array([x_guess,y_guess,R_guess,R_guess,0]);
        #print(guess)
        lower_bounds = [min(x),min(y),0,0,0]
        upper_bounds = [max(x),max(y),max(max(x),max(y))/2,max(max(x),max(y))/2,2*np.pi]
        bounds = (lower_bounds,upper_bounds)
    except:
        raise Exception(points.shape)
    return(optimize.least_squares(fun=funcEclipse,x0=guess, \
                                  xtol = 1e-12,bounds=bounds,args=(x,y)).x)





# generate a circle
def generateCircle(xyr,n=100):
    theta = np.linspace(0,2*np.pi,n);
    x = xyr[0] + xyr[2] * np.cos(theta)
    y = xyr[1] + xyr[2] * np.sin(theta)
    return(x,y)

# compute the arctangent from xy coordinates
def atan(x,y,x0=0,y0=0):
    n = len(x);
    x = x-x0;
    y = y-y0;
    if n != len(y):
        raise('The length of x and length of y should be the same!')
    
    theta = np.zeros(n);
    for i in range(n):
        if x[i] == 0 and y[i] == 0:
            theta[i] = 0
        elif x[i] == 0 and y[i] > 0:
            theta[i] = np.pi/2
        elif x[i] == 0 and y[i] < 0:
            theta[i] = np.pi*3/2
        elif y[i] == 0 and x[i] > 0:
            theta[i] = 0
        elif y[i] == 0 and x[i] < 0:
            theta[i] = np.pi
            
        elif x[i] >= 0 and y[i] >= 0:
            theta[i] = np.arctan(y[i]/x[i]);
        elif x[i] >= 0 and y[i] < 0:
            theta[i] = 2 * np.pi - np.arctan(abs(y[i]/x[i]));
        elif x[i] < 0 and y[i] >= 0:
            theta[i] = np.pi - np.arctan(abs(y[i]/x[i])) 
        elif x[i] < 0 and y[i] < 0:
            theta[i] = np.pi + np.arctan(abs(y[i]/x[i])) 
    return(theta)


# low-pass filter
from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# averaging filter
def avgFilter(data,window):
    data1 = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window-1:
            data1[i] = (sum(data[(i-window+1):])+sum(data[0:i+1]))/window
            #print(i)
            #print(data[(i-window+1):],data[0:i+1])
            
        else:
            data1[i] = np.mean(data[(i-window):(i)])
            #print(i-window,i)
            #print(data[i-window:i])
    return(data1)






# test
if __name__ == '__main__':
    workpath = '../test';
    filename = 'ts3_0000012622.tif';
    thre = 127;
    iteration = [1,5,4]
    showresult = False;
    #window = [18,20,25]
    window = [20]

    filename = workpath +"/" + filename
    cha1,contours = findOilBlob(filename,iterations=iteration,showresult=showresult);
    #cv2.drawContours(cha1, contours, contourIdx = -1, color=(255,0,0), \
    #                 thickness=4)
    #imshow(cha1)
    
    oilcontour = contours[1][:,0,:].astype(float);
    oilcontour0 = np.copy(oilcontour)
    
    '''
    # low pass filter
    oilcontour[:,0] = butter_lowpass_filter(oilcontour[:,0], cutoff, fs, order)
    oilcontour[:,1] = butter_lowpass_filter(oilcontour[:,1], cutoff, fs, order)
    '''
    
    # filter and subsample
    freqratio = 3;
    # averaging filter
    oilcontour[:,0] = avgFilter(oilcontour[:,0], int(freqratio))
    oilcontour[:,1] = avgFilter(oilcontour[:,1], int(freqratio))
    # subsample
    ind = np.arange(0,int(len(oilcontour)/freqratio));
    oilcontour = oilcontour[ind*freqratio]


    # plot the oil blob
    plt.figure()
    plt.plot(oilcontour[:,0],oilcontour[:,1],'s',label='subsampled oil contour')
    plt.plot(oilcontour0[:,0],oilcontour0[:,1],'rs',label='oil contour')
    #plt.axis('square')
    #plt.title('oil blob')
    
    # find the ecliptical fit
    eclps = findEclipse(oilcontour)
    
    theta = np.linspace(0,2*np.pi,60)
    X = eclps[2]*np.cos(theta);
    Y = eclps[3]*np.sin(theta);
    x = eclps[0] + X*np.cos(eclps[4])-Y*np.sin(eclps[4])
    y = eclps[1] + X*np.sin(eclps[4])+Y*np.cos(eclps[4])
    plt.plot(x,y,'r-',label='fitted eclipse')
    plt.axis('square')
    plt.title('oil blob')
    plt.legend()
    
    eclpsxy = np.array([x,y]).transpose()
    

    for lp0 in window:
        
        curvatureradius = findCurvatureRadius(oilcontour,window=lp0);
        curvatureeclps = findCurvatureRadius(eclpsxy,window=lp0);
        
        tmpcha = np.copy(cha1)
        for i in range(len(curvatureradius[:,0])):
            #if np.mod(i,10)!=0:
            #    continue
            x,y = generateCircle(curvatureradius[i,:],300)
            try:
                tmpcha[y.astype(int),x.astype(int)] = 0;
            except:
                pass
        plt.imsave(workpath+'/pore'+str(lp0)+'.jpg',tmpcha)
        
        fig = plt.figure()
        plt.plot(eclpsxy[:,0],eclpsxy[:,1])
        for i in range(len(curvatureeclps[:,0])):
            #if np.mod(i,10)!=0:
            #    continue
            x,y = generateCircle(curvatureeclps[i,:],300)
            try:
                #tmpcha[y.astype(int),x.astype(int)] = 0;
                plt.plot(x,y)
            except:
                pass
        plt.axis('square')
        plt.savefig(workpath+'/eclipse'+str(lp0)+'.jpg')
        
        
        
        
        
            
        # compute the phase angle of each point on the oil-water interface
        theta = atan(oilcontour[:,0],oilcontour[:,1], \
                     (max(oilcontour[:,0])+min(oilcontour[:,0]))/2, \
                     (max(oilcontour[:,1])+min(oilcontour[:,1]))/2)
        
        # compute the phase angle of each point on the oil-water interface
        thetaeclps = atan(eclpsxy[:,0],eclpsxy[:,1], \
                     (max(eclpsxy[:,0])+min(eclpsxy[:,0]))/2, \
                     (max(eclpsxy[:,1])+min(eclpsxy[:,1]))/2)
   
        '''
        plt.figure()
        plt.plot(curvatureradius[:,2],'rs-')
        #plt.plot(y,'rs-')
        #plt.plot(yavg,'rs-')
        plt.xlabel('index')
        plt.ylabel('curvature radius')
        plt.title('radius')
        '''
        
        thetaeclps1 = np.roll(thetaeclps,int((freqratio+lp0)/2))
        plt.figure()
        plt.plot(theta*180/np.pi,curvatureradius[:,2],'rs',label='oil blob')
        plt.plot(thetaeclps1*180/np.pi,curvatureeclps[:,2],'b+',label='fitted eclipse')
        #kappa = 2/((eclps[2]*np.sin(theta+eclps[4]))**2+(eclps[3]*np.sin(theta+eclps[4]))**2)**(3/2)
        R = ((eclps[2]*np.sin(theta+eclps[4]))**2+(eclps[3]*np.cos(theta+eclps[4]))**2)**(3/2) / eclps[2] / eclps[3]
        theta1 = np.roll(theta,int((freqratio-lp0)/2))
        plt.plot(theta1*180/np.pi, R,'b-',label='analitycal eclipse')
        
        plt.xlabel('phase angle')
        plt.ylabel('curvature radius')
        plt.title('radius')
        plt.legend()
        
        
        plt.figure()
        plt.hist(curvatureradius[:,2],bins=10)
        plt.title('radius')
        
        
        
        plt.figure()
        # semimajor axis coordinate
        x=oilcontour[:,0]*np.cos(eclps[4]) - oilcontour[:,1]*np.sin(eclps[4]);
        plt.plot(x,curvatureradius[:,2],'bs-',label='curvature radius')
        plt.plot(x,theta*180/np.pi,label='phase angle')
        plt.xlabel('semi-major axis coordinate')
        #plt.ylabel()
        plt.legend()

        plt.figure()
        # semimajor axis coordinate
        x=oilcontour[:,0]*np.cos(eclps[4]) - oilcontour[:,1]*np.sin(eclps[4]);
        plt.loglog((1+((eclps[2]/eclps[3])**2-1)*x**2),curvatureradius[:,2],'bs-',label='curvature radius')
        plt.xlabel('semi-major axis coordinate')
        plt.legend()
        