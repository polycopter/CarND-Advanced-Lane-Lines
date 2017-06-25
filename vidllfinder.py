import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
import os
# needed to process video (*.mp4 files)
from moviepy.editor import VideoFileClip


def unpack(line):
   x1 = line[0][0]
   y1 = line[0][1]
   x2 = line[0][2]
   y2 = line[0][3]
   return x1,y1,x2,y2
   
def slope(line):
   x1,y1,x2,y2 = unpack(line)
   if (x1==x2):
      return 100000000 # arbitrary big number
   return (y2 - y1)/(x2 - x1)
   
# assumes a coordinate system that doesn't go out to 100000000
def smallest_x(lines):
   smallest = 100000000
   for line in lines:
      x,y,x2,y2 = unpack(line)
      if x < smallest:
         smallest = x
      if x2 < smallest:
         smallest = x2
   return smallest

# assumes a coordinate system that doesn't go out to 100000000
def smallest_y(lines):
   smallest = 100000000
   for line in lines:
      x,y,x2,y2 = unpack(line)
      if y < smallest:
         smallest = y
      if y2 < smallest:
         smallest = y2
   return smallest

# assumes a coordinate system that doesn't go negative
def largest_x(lines):
   largest = 0
   for line in lines:
      x,y,x2,y2 = unpack(line)
      if x > largest:
         largest = x
      if x2 > largest:
         largest = x2
   return largest

# assumes a coordinate system that doesn't go negative
def largest_y(lines):
   largest = 0
   for line in lines:
      x,y,x2,y2 = unpack(line)
      if y > largest:
         largest = y
      if y2 > largest:
         largest = y2
   return largest
   
def extend_to_bottom(img_bottom,x1, y1, x2, y2):   
   m = slope([[x1,y1,x2,y2]])
   b = y1 - m*x1
   x_at_bottom = (img_bottom - b)/m
   # debug
   # print(m,b,x1,y1,x2,y2)
   if y1 > y2:
      return [int(x_at_bottom),img_bottom,x2,y2]
   
   return [x1,y1,int(x_at_bottom),img_bottom]

def remove_outliers(lines, roi_top):
   ret_lines = []
   for i in range(len(lines)):
      line = lines[i][0]
      #slope = lines[i][1]
      #print( slope, line )
      # 
      if line[0][1] > roi_top and line[0][3] > roi_top:
         ret_lines.append(line)
      
   return ret_lines
   
# this [kludge?] is based on the assumption that the lane-line
# to the car's left has positive slope, the other negative
# (except reversed, due to 'upside-down' coordinate system)
def connect_lines(lines, img_bottom, roi_top):
   line_left = None
   left_lines = []
   line_right = None
   right_lines = []
   for line in lines:
      #print(line)
      m = slope(line)
      if m < -0.5:
         left_lines.append((line,m))
      elif m > 0.5:
         right_lines.append((line,m))
   
   left_lines = remove_outliers(left_lines, roi_top)
   if len(left_lines) > 0:
      x1left = smallest_x(left_lines)
      y1left = largest_y(left_lines)
      x2left = largest_x(left_lines)
      y2left = smallest_y(left_lines)
      line_left = [extend_to_bottom(img_bottom,x1left, y1left, x2left, y2left)]
   
   right_lines = remove_outliers(right_lines, roi_top)
   if len(right_lines) > 0:
      x1right = smallest_x(right_lines)
      y1right = smallest_y(right_lines)
      x2right = largest_x(right_lines)
      y2right = largest_y(right_lines)
      line_right = [extend_to_bottom(img_bottom,x1right, y1right, x2right, y2right)]
   
   if line_left is not None and line_right is not None:
      lines_out = [line_left, line_right]
      return lines_out
      
   return [] # hough found no "good" lines to connect/extend

# below here, mostly stolen from the hough quiz

def pipeline(fname_in, fname_out):
   # Read in the image
   image = mpimg.imread(fname_in)
   rgb_img = process_image(image)
    
   #save a copy
   # plt.imshow() shows red lines, but if you don't do
   # this cvtColor step, they appear blue in the saved image
   out_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR)
   cv2.imwrite(fname_out, out_img)

def process_image(image):
   # NOTE: The output you return 
   # should be a color image (3 channel) for processing video below
   # you should return the final output (image with lines are drawn on lanes)

   # Here we convert to 0,255 bytescale
   gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

   # Define a kernel size and apply Gaussian smoothing
   kernel_size = 5
   blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

   # Define our parameters for Canny and apply
   low_threshold = 50
   high_threshold = 150
   edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

   # Next we'll create a masked edges image using cv2.fillPoly()
   mask = np.zeros_like(edges)   
   ignore_mask_color = 255   

   # This time we are defining a four sided polygon to mask
   imshape = image.shape
   top_y = imshape[0]/2+50
   vertices = np.array([[(30,imshape[0]), (imshape[1]/2 - 20, top_y), (20+imshape[1]/2, top_y), (imshape[1]-30,imshape[0])]], dtype=np.int32)
   cv2.fillPoly(mask, vertices, ignore_mask_color)
   masked_edges = cv2.bitwise_and(edges, mask)

   # Define the Hough transform parameters
   # Make a blank the same size as our image to draw on
   rho = 2 # distance resolution in pixels of the Hough grid
   theta = np.pi/180 # angular resolution in radians of the Hough grid
   threshold = 15    # minimum number of votes (intersections in Hough grid cell)
   min_line_length = 40 #minimum number of pixels making up a line
   max_line_gap = 20    # maximum gap in pixels between connectable line segments
   line_image = np.copy(image)*0 # creating a blank to draw lines on

   # Run Hough on edge detected image
   # Output "lines" is an array containing endpoints of detected line segments
   lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

   # connect the dashed lines
   clines = connect_lines(lines, imshape[0], top_y)
   # Iterate over the output "lines" and draw lines on a blank image
   for line in clines:
      for x1,y1,x2,y2 in line:
         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

   # Create a "color" binary image to combine with line image
   color_edges = np.dstack((edges, edges, edges)) 

   # Draw the lines on the edge image
   lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

   # draw the lines on the original image
   final_img = cv2.addWeighted(image, 0.8, line_image, 1, 0)

   # btw, this plt.imshow doesn't work on my Linux Mint or OS X Sierra, 
   # however it seems to work within the jupyter environment
   #plt.imshow(lines_edges)

   #return final_img
   return lines_edges
   

if __name__ == '__main__':
   test_img = os.listdir('test_images')

   for fname in test_img:
      print('processing', fname)
      pipeline('test_images/'+fname, 'output_images/'+fname)
   
   projvid_output = 'output_videos/projvid.mp4'
   clip1 = VideoFileClip("project_video.mp4")
   basic_clip = clip1.fl_image(process_image)
   # %time 
   basic_clip.write_videofile(projvid_output, audio=False)

   challenge_output = 'output_videos/challenge.mp4'
   clip2 = VideoFileClip('challenge_video.mp4')
   tough_clip = clip2.fl_image(process_image)
   # %time 
   tough_clip.write_videofile(challenge_output, audio=False)

   harder_challenge_output = 'output_videos/harder-challenge.mp4'
   clip2 = VideoFileClip('harder_challenge_video.mp4')
   tougher_clip = clip2.fl_image(process_image)
   # %time 
   tougher_clip.write_videofile(harder_challenge_output, audio=False)

