"""
Process the data from PPT ID: 20191107

Created on 2020-05-29 10:43

@Author: Ningyu Wang

"""

from findCurvatureRadii import os,np,cv2,plt
from findCurvatureRadii import removeConcave,subSample,findOilBlob,findCurvatureRadius

from imshow import imshow


if __name__ == '__main__':

    cv2.destroyAllWindows()
    plt.close('all')
    '''
    cut the image, use only the left hand side half
    '''
    # working directory
    sourcepath_cut = '../../../../my_publication/EOR/ATCE2020/fig/exp1_ts3_0_4spf';
    targetpath_cut = sourcepath_cut +'/cut';

    from cut import cutall
    cutall(x=[0.3,0.7],y=[0.25,0.75],sourcepath=sourcepath_cut,targetpath=targetpath_cut)


    # working directory
    #sourcepath = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_1fps/cut';
    sourcepath = targetpath_cut
    targetpath = sourcepath +'/../result';

    # parameters
    thre = 127;
    #iteration = [1,5,4]
    #iteration = [1,2,1]
    iteration = [0,0,0]
    showresult = True;
    dotsize = 2;
    subsampledistance = 3;


    # average filter:
    # average this number of points to generate a point
    freqratio = 2;

    # number of points used to fit the circle
    window = [9]

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
    maxradiustotalp = 0; # positive side
    minradiustotalp = 0;
    maxradiustotaln = 0; # negative side
    minradiustotaln = 0;


    print('Processing...')
    # This is for debuging. Run only certain number of images.
    count = 0;
    for ite in enumerate(files):
        print(' '+ite[1])

        filename = sourcepath+'/'+ite[1];

        # get the raw contour
        cha1,contours,imorig,contoursorig = findOilBlob(filename,color='red',iterations=iteration,showresult=showresult);

        # shrink unwanted dimension
        oilcontour = contours[1][:,0,:].astype(float);

        # subsample
        print('size before subsample: ',oilcontour.shape)
        oilcontour = subSample(oilcontour,distancetol=subsampledistance)
        print('size after subsample: ',oilcontour.shape)

        '''
        # remove concave sections
        oilcontour = removeConcave(oilcontour)
        print('size after removal of concave points: ',oilcontour.shape)
        '''

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

        maxradiusp = max(curvatureradius[:,2])
        minradiusp = min(curvatureradius[:,2])

        maxradiusn = max(curvatureradius[:,2])
        minradiusn = min(curvatureradius[:,2])


        maxradiusp = 80;
        minradiusp = 0;

        maxradiusn = -80;
        minradiusn = 0;

        # See what is the range of the cavature radius
        # positive side
        if maxradiustotalp < maxradiusp:
            maxradiustotalp = np.copy(maxradiusp)
        if minradiustotalp > minradiusp:
            minradiustotalp = np.copy(minradiusp)
        # negative side
        if maxradiustotaln > maxradiusn:
            maxradiustotaln = np.copy(maxradiusn)
        if minradiustotaln < minradiusn:
            minradiustotaln = np.copy(minradiusn)

        numcontourpoint = curvatureradius.shape[0]
        curvatureradiusplot = np.zeros(numcontourpoint)

        Nx,Ny,__ = imorig.shape
        for lp1 in range(numcontourpoint):
            if curvatureradius[lp1,2] > 0:
                # positive side
                curvatureradiusplot[lp1] = \
                    (curvatureradius[lp1,2]-minradiusp) / (maxradiusp-minradiusp) * 256
                # coordinate index
                indy = int(oilcontour[lp1,0])
                indx = int(oilcontour[lp1,1])
                # color
                imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,[0,1]] = 0
                imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,2] = curvatureradiusplot[lp1]
            else:
                # negative side
                curvatureradiusplot[lp1] = \
                    (curvatureradius[lp1,2]-minradiusn) / (maxradiusn-minradiusn) * 256
                # coordinate index
                indy = int(oilcontour[lp1,0])
                indx = int(oilcontour[lp1,1])
                # color
                imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,[1,2]] = 0
                imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,0] = curvatureradiusplot[lp1]

        imshow(imorig,showresult)
        savename = targetpath+'/'+ite[1]
        cv2.imwrite(savename,imorig)

        '''
        count = count+1
        if count > 5:
            break
        '''

    print(maxradiustotalp,minradiustotalp)
    print(maxradiustotaln,minradiustotaln)

    # plot the curvature radius of the last image
    plt.figure()
    plt.plot(curvatureradius[:,2],'sr-')

    # plot the color bar for the curvature radius
    tmp = np.linspace(1, 0, 256)
    fig = plt.figure()
    n = len(tmp)
    img = np.zeros((n*2,20,3))
    for ite in range(20):
        img[0:n,ite,0] = tmp
        img[-1:-n-1:-1,ite,2] = tmp
    h = plt.imshow(img)

    plt.ylim(511,0)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    t1p = round(minradiusp*220/1280*8);
    t3p = round(maxradiusp*220/1280*8);
    t2p = round((t1p+t3p)/2);
    t1n = round(minradiusn*220/1280*8);
    t3n = round(maxradiusn*220/1280*8);
    t2n = round((t1n+t3n)/2);
    plt.yticks(ticks=[511,383,255,127,0], \
               labels=[t3n,t2n,0,t2p,t3p])
    plt.title(r'$\mu m$')
