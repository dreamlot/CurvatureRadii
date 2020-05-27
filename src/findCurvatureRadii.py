# -*- coding: utf-8 -*-
"""
This program is to find the curvature radii along the oil blob perimeter.
It studies only the oil blob on the left hand side.
The figure will be cut in half and the right hand side half image would
be abandoned.


Created on Thu Feb 27 21:27:19 2020

@author: ningyu
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# This implies a code from internet. Reference see inside.
from findCurvatureRadii2 import findCurvatureR

# show image
from imshow import imshow


# Remove straight section. Straight lines have infinite curvature radius.
def removeConcave(points,tol=1e-13):
    m,n = points.shape;

    # Check each three adjacent points from the last in the list.
    # If it is a concaving section, remove the middle point.
    # In this way, when a point is removed, the two remaining points from the
    # last step enters the next iteration.


    # First, determine if the points are in clockwise or counter- order.
    FLAG_CLDIR= 1; # conter-clockwise
    dind = 3
    r1 = points[dind,:] - points[0,:]
    r2 = points[dind*2,:] - points[dind,:]
    if np.cross(r1,r2) < 0:
        # clockwise
        FLAG_CLDIR = -1;

    # Then, remove the concave points
    for ite in range(m-1,1,-1):
        r1 = points[ite-2,:] - points[ite-1,:]
        r2 = points[ite-1,:] - points[ite,:]

        # Use cross product to determine if the vector between points
        # is rotating in the clockwise direction or the opposite.
        # If it is different from the overall direction,
        # remove the middle point.
        if np.cross(r1,r2) * FLAG_CLDIR < 0:
            #print(ite,r1,r2)
            points = np.delete(points,ite-1,0)


    # At last, check the two sections containing the first and last points.
    r1 = points[1,:] - points[0,:]
    r2 = points[0,:] - points[-1,:]
    if 1-abs( np.dot(r1,r2) / np.linalg.norm(r1) / np.linalg.norm(r2) ) < tol:
        points = np.delete(points,0,0)
    r1 = points[0,:] - points[-1,:]
    r2 = points[-1,:] - points[-2,:]
    if 1-abs( np.dot(r1,r2) / np.linalg.norm(r1) / np.linalg.norm(r2) ) < tol:
        points = np.delete(points,-1,0)
        
    return(points)


# Subsample the regions of high point density
def subSample(points,distancetol=5):
    m,n = points.shape;
    '''
    # if three points are adjacent, remove the middle point
    for ite in range(m-1,1,-1):
        r = points[ite,:] - points[ite-2,:]
        if np.linalg.norm(r,2) < distancetol:
            np.delete(points,ite-1,0)
    '''
    # if two points are adjacent, remove the earlier point
    for ite in range(m-1,0,-1):
        r = np.linalg.norm(points[ite,:] - points[ite-1,:],2)
        #print(r)
        if r < distancetol:
            #print('delete',ite)
            points = np.delete(points,ite-1,0)

    return(points)



# Interpolate the contour for long straight sections
def interpolateContour(points,insert=1,distancetol=5):
    points = np.asarray(points)
    n = len(points[:,0])
    '''
    # first, deal with the gap between the first and last points
    d = np.linalg.norm(points[0,:]-points[-1,:]);
    if d > distancetol:
        a = points[-1,:];
        b = points[0,:]
        for j in range(insert,0,-1):
            x = a[0] * (insert-j) / insert + b[0] * j / insert
            y = a[1] * (insert-j) / insert + b[1] * j / insert
            points = np.insert(points,n,[x,y],axis=0)
    '''

    # then, deall with other points
    for i in range(n-1,0,-1):
        d = np.linalg.norm(points[i,:]-points[i-1,:]);
        if d > distancetol:
            a = points[i-1,:]
            b = points[i,:]
            for j in range(insert,0,-1):
                x = a[0] * (insert-j) / insert + b[0] * j / insert
                y = a[1] * (insert-j) / insert + b[1] * j / insert
                points = np.insert(points,i,[x,y],axis=0)
    return(points)



# Get the oil blob perimeter
# Returns the grayscale image and a 2D array of xy coordinates
# of points on the contour.
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
    imgrayorig = imgray.copy()
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
    #ret, thresh = cv2.threshold(dilation, 127, 255, 0)
    ret, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_OTSU)
    imshow(thresh,showresult)
    '''
    # The cv2.findContours() function removed the first output.
    tmpim, contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
                                              mode=cv2.CHAIN_APPROX_SIMPLE)
    '''
    contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
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

    return (imgray,contours,imgrayorig)

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
# return x,y,r of the inscribed circle at each point
# NW modify on May22-2020 to locate the window around the point
# instead of after it.
def findCurvatureRadius(points,window=5):
    # number of points
    N = points.shape[0];

    # half window length used to fit the circle
    n = int(window/2);




    #print(N)
    # fitted circle
    result = np.zeros([N,3]);

    try:
        for i in range(N):
            if i < n+1:
                indx = np.concatenate((np.arange(N+i-n,N), np.arange(0,i+n+1)))
            elif i + n >= N:
                indx = np.concatenate((np.arange(i-n,N), np.arange(0,i+n-N)))
            else:
                indx = np.arange((i-n),(i+n+1))

            #result[i,:] = findCircle(points[indx,:]);
            x = points[indx,0]
            y = points[indx,1]
            result[i,:] = findCurvatureR(x,y)
    except:
        print(i,n,N)
        print(N+i-n,i+n+1)
        print(i-n,i+n-N)


    return( result )



# Function of a ellipse
def funcEllipse(para,x,y):
    return( (((x-para[0])*np.cos(para[4])+(y-para[1])*np.sin(para[4]))/para[2])**2 \
             + ((-(x-para[0])*np.sin(para[4])+(y-para[1])*np.cos(para[4]))/para[3])**2 \
             - 1 )
    #return( np.sqrt((x-para[0])**2 + (y-para[1])**2) - para[2])

'''
# from a set of x y values, fit a ellipse
input:
    points: n x 2, np.array
output:
        : list:
            x0:     x coordinate of ellipse
            y0:     y coordinate of ellipse
            a:      semi-major axis
            b:      semi-minor axis
            theta:  tilt angle
'''
def findEllipse(points):
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
    return(optimize.least_squares(fun=funcEllipse,x0=guess, \
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
def test1():
    #workpath = '../test';
    workpath = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_1fps';
    #filename = 'ts3_0000010081.tif';
    #filename = 'ts3_0000012692.tif';
    #filename = 'ts3_0000012718.tif';
    #filename = 'ts3_0000010072.tif';
    #filename = 'ts3_0000010158.tif';
    #filename = 'ts3_0000012622.tif';
    filename = 'ts3_001005.tif';
    thre = 127;
    iteration = [1,5,4]
    showresult = False;


    # average filter:
    # average this number of points to generate a point
    freqratio = 3;

    # number of points used to fit the circle
    #window = [18,20,25]
    window = [20]
    #window = [15]


    filename = workpath +"/" + filename
    cha1,contours,__ = findOilBlob(filename,iterations=iteration,showresult=showresult);
    #cv2.drawContours(cha1, contours, contourIdx = -1, color=(255,0,0), \
    #                 thickness=4)
    #imshow(cha1,showresult)

    # shrink unwanted dimension
    oilcontour = contours[1][:,0,:].astype(float);

    oilcontour0 = np.copy(oilcontour)

    # save a copy of the original image
    oilorg = np.copy(oilcontour)

    # interpolate straight section
    #oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=30)
    oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=10)


    plt.figure()
    plt.plot(oilcontour[:,0],oilcontour[:,1],'bs-')
    plt.plot(oilorg[:,0],oilorg[:,1],'r+')


    '''
    # low pass filter
    oilcontour[:,0] = butter_lowpass_filter(oilcontour[:,0], cutoff, fs, order)
    oilcontour[:,1] = butter_lowpass_filter(oilcontour[:,1], cutoff, fs, order)
    '''

    # filter and subsample
    # averaging filter
    oilcontour[:,0] = avgFilter(oilcontour[:,0], int(freqratio))
    oilcontour[:,1] = avgFilter(oilcontour[:,1], int(freqratio))
    # subsample
    ind = np.arange(0,int(len(oilcontour)/freqratio));
    oilcontour = oilcontour[ind*freqratio]


    # plot the oil blob
    plt.figure()
    plt.plot(oilcontour0[:,0],oilcontour0[:,1],'rs',label='oil contour')
    plt.plot(oilcontour[:,0],oilcontour[:,1],'s',label='subsampled oil contour')
    #plt.axis('square')
    #plt.title('oil blob')

    # find the ecliptical fit
    ellps = findEllipse(oilcontour)

    theta = np.linspace(0,2*np.pi,len(oilcontour[:,0]))
    X = ellps[2]*np.cos(theta);
    Y = ellps[3]*np.sin(theta);
    x = ellps[0] + X*np.cos(ellps[4])-Y*np.sin(ellps[4])
    y = ellps[1] + X*np.sin(ellps[4])+Y*np.cos(ellps[4])

    plt.plot(x,y,'r-',label='fitted ellipse')
    plt.axis('square')
    plt.title('oil blob',fontsize=12)
    plt.legend(prop={'size': 12})
    plt.xlabel('pixels',fontsize=12)
    plt.ylabel('pixels',fontsize=12)

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
        plt.savefig(workpath+'/ellipse'+str(lp0)+'.jpg')






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

        plt.figure()
        thetaeclps1 = np.roll(thetaeclps,int((freqratio+lp0)/2))
        plt.plot(theta*180/np.pi,curvatureradius[:,2],'rs',label='oil blob')
        plt.plot(thetaeclps1*180/np.pi,curvatureeclps[:,2],'b+',label='fitted ellipse')
        #kappa = 2/((ellps[2]*np.sin(theta+ellps[4]))**2+(ellps[3]*np.sin(theta+ellps[4]))**2)**(3/2)
        R = ((ellps[2]*np.sin(theta+ellps[4]))**2+(ellps[3]*np.cos(theta+ellps[4]))**2)**(3/2) / ellps[2] / ellps[3]
        theta1 = np.copy(theta)
        theta1 = np.roll(theta1,int((freqratio-lp0)/2))
        #plt.plot(theta1*180/np.pi, R,'b-',label='analitycal ellipse')

        plt.xlabel('phase angle')
        plt.ylabel('curvature radius')
        plt.title('radius')
        plt.legend()

        '''
        plt.figure()
        plt.hist(curvatureradius[:,2],bins=10)
        plt.title('radius')
        '''


        plt.figure()
        # semimajor axis coordinate
        x=(oilcontour[:,0]-ellps[0])*np.cos(ellps[4]) - (oilcontour[:,1]-ellps[1])*np.sin(ellps[4]);
        x1= np.roll(x,int((-freqratio+lp0)/2))
        plt.plot(x1,curvatureradius[:,2],'bs-',label='curvature radius')
        plt.plot(x1,theta*180/np.pi,label='phase angle')
        plt.xlabel('semi-major axis coordinate')
        #plt.ylabel()
        plt.legend()

        '''
        plt.figure()
        # semimajor axis coordinate
        x=oilcontour[:,0]*np.cos(ellps[4]) - oilcontour[:,1]*np.sin(ellps[4]);
        plt.loglog((1+((ellps[2]/ellps[3])**2-1)*x**2),curvatureradius[:,2],'bs-',label='curvature radius')
        plt.xlabel('semi-major axis coordinate')
        plt.legend()
        '''

        plt.figure()
        x = (oilcontour[:,0]-ellps[0])*np.cos(ellps[4]) - (oilcontour[:,1]-ellps[1])*np.sin(ellps[4]);
        y = (oilcontour[:,0]-ellps[0])*np.sin(ellps[4]) + (oilcontour[:,1]-ellps[1])*np.cos(ellps[4]);
        plt.plot(x,y,'s-')


if __name__ == '__main__':
    '''
    cut the image, use only the left hand side half
    '''
    # working directory
    sourcepath_cut = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_2fps';
    targetpath_cut = sourcepath_cut +'/cut';

    from cut import cutall
    #cutall(y=[0,0.5],sourcepath=sourcepath_cut,targetpath=targetpath_cut)


    # working directory
    #sourcepath = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_1fps/cut';
    sourcepath = targetpath_cut
    targetpath = sourcepath +'/../result';

    # parameters
    thre = 127;
    iteration = [1,5,4]
    showresult = False;
    dotsize = 3;
    subsampledistance = 9;


    # average filter:
    # average this number of points to generate a point
    freqratio = 2;

    # number of points used to fit the circle
    window = [19]

    # load the files in the source directory
    # @Vaibhav
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    from os.path import isfile, join
    files = [f for f in os.listdir(sourcepath) if isfile(join(sourcepath, f))]

    # prepare the output directory
    try:
        os.mkdir(targetpath)
    except FileExistsError:
        pass


    # record the max and min of curvature radius
    maxradiustotal = 0;
    minradiustotal = 0;


    print('Processing...')
    # This is for debuging. Run only certain number of images.
    count = 0;
    for ite in enumerate(files):
        print(' '+ite[1])

        filename = sourcepath+'/'+ite[1];

        # get the raw contour
        cha1,contours,imorig = findOilBlob(filename,iterations=iteration,showresult=showresult);

        # shrink unwanted dimension
        oilcontour = contours[1][:,0,:].astype(float);
        '''
        # interpolate straight section
        oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=10)
        '''
        # subsample
        '''
        ind = np.arange(0,int(len(oilcontour)/freqratio));
        oilcontour = oilcontour[ind*freqratio]
        '''
        print('size before subsample: ',oilcontour.shape)
        oilcontour = subSample(oilcontour,distancetol=subsampledistance)
        print('size after subsample: ',oilcontour.shape)
        # remove concave sections
        oilcontour = removeConcave(oilcontour)
        print('size after removal of concave points: ',oilcontour.shape)


        '''
        # averaging filter
        oilcontour[:,0] = avgFilter(oilcontour[:,0], int(freqratio))
        oilcontour[:,1] = avgFilter(oilcontour[:,1], int(freqratio))
        '''


        # compute curvature
        curvatureradius = findCurvatureRadius(oilcontour,window=window[0]);

        # recover the three channels of original figure
        imorig = cv2.cvtColor(imorig,cv2.COLOR_GRAY2BGR)


        # write points into image

        #maxradius = max(curvatureradius[:,2])
        #minradius = min(curvatureradius[:,2])

        maxradius = 200;
        minradius = 100;
        
        # See what is the range of the cavature radius
        if maxradiustotal < maxradius:
            maxradiustotal = np.copy(maxradius)
        if minradiustotal > minradius:
            minradiustotal = np.copy(minradius)
        
        numcontourpoint = curvatureradius.shape[0]
        curvatureradiusplot = (curvatureradius[:,2]-minradius) / (maxradius-minradius) * 256
        Nx,Ny,__ = imorig.shape
        for lp1 in range(numcontourpoint):
            # coordinate index
            indy = int(oilcontour[lp1,0])
            indx = int(oilcontour[lp1,1])
            # color
            imorig[indx-2:indx+3,indy-2:indy+3,[0,1]] = 0
            imorig[indx-2:indx+3,indy-2:indy+3,2] = curvatureradiusplot[lp1]

        imshow(imorig,showresult)
        savename = targetpath+'/'+ite[1]
        cv2.imwrite(savename,imorig)
        
        '''
        count = count+1
        if count > 5:
            break
        '''
        
    print(maxradiustotal,minradiustotal)
    
    
    plt.figure()
    plt.plot(curvatureradius[:,2],'sr-')