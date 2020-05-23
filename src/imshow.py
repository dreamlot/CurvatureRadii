'''
# show image in OpenCV

2020-05-14 23:01
@ Ningyu Wang

'''
import cv2
import threading

import matplotlib.pyplot as plt


# show image
def imshowfunc(img,name='result',x=800,y=600):
    cv2.namedWindow(str(name),cv2.WINDOW_NORMAL);
    cv2.imshow(str(name),img);
    cv2.resizeWindow(str(name), x,y);
    cv2.waitKey(0);
    #cv2.destroyAllWindows();

def imshow(img,showresult=True,name='result',x=800,y=600):
    if not showresult:
        return
    #imshowfunc(img,name,x,y)
    '''
    plt.figure()
    plt.imshow(img)
    plt.title(name)
    '''

    try:
        x = threading.Thread( target=imshowfunc, args=(img,name,x,y) );
        x.start()
    except:
        pass
    
