import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# read in a set of calibr8n imgs
imgs = glob.glob('camera_cal/calibration*.jpg')
#mpimg.imsave('test.jpg',img)

objpt = []
imgpt = []

xdim = 9
ydim = 6
# prep onj pts, e.g. (0,0,0), (1,0,0), ... 
objp = np.zeros((xdim*ydim,3), np.float32)
objp[:,:2] = np.mgrid[0:xdim,0:ydim].T.reshape(-1,2)

for fname in imgs:
    img = mpimg.imread(fname)
    # greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(grey, (xdim, ydim), None)

    if ret == True:
        imgpt.append(corners)
        objpt.append(objp)

#        img = cv2.drawChessboardCorners(img, (xdim, ydim), corners, ret)
#        mpimg.imsave('corners.jpg', img)

    else:
        print(fname)
        mpimg.imsave(fname[0:-3] + '-ng.jpg',grey)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpt, imgpt, grey.shape[::-1], None, None)
# read test image
img = mpimg.imread('camera_cal/calibration1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
mpimg.imsave('undist.jpg', dst)

