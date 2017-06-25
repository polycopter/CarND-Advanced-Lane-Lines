import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# read in a set of calibr8n imgs
# (NOTE hard-coded folder name & relative location)
imgs = glob.glob('camera_cal/calibration*.jpg')

#for debug:
#mpimg.imsave('test.jpg',img)

objpt = []
imgpt = []

# shape of the checkerboard
xdim = 9
ydim = 6

# prep obj pts, e.g. (0,0,0), (1,0,0), ... 
objp = np.zeros((xdim*ydim,3), np.float32)
objp[:,:2] = np.mgrid[0:xdim,0:ydim].T.reshape(-1,2)

# process the calibration images
for fname in imgs:
    img = mpimg.imread(fname) # NOTE: mpimg --> BGR, cv2 --> RGB
    # greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(grey, (xdim, ydim), None)

    if ret == True:
        imgpt.append(corners)
        objpt.append(objp)

# for debug:
#        img = cv2.drawChessboardCorners(img, (xdim, ydim), corners, ret)
#        mpimg.imsave('corners.jpg', img)

    else:
        # save a copy of each image that cv2 failed to find the corners in,
        # just to make it easy to view them for edification
        print(fname)
        mpimg.imsave(fname[0:-3] + '-ng.jpg',grey) # 'ng' as in "no good"

# generate the calibration transformations
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpt, imgpt, grey.shape[::-1], None, None)

# read test calibration image 
# (NOTE hard-coded folder names & relative location)
img = mpimg.imread('camera_cal/calibration1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
mpimg.imsave('output_images/undist.jpg', dst)

# read test image (NOTE hard-coded relative folder name & location)
img = mpimg.imread('test_images/test4.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
mpimg.imsave('test4-undist.jpg', dst)



