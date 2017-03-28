from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import numpy
import scipy
from scipy import ndimage
import sys
from matplotlib import pyplot as plt
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage


def rgb2heat(rgb):
    '''Return the heatmap version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    heat = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return heat/255.

# Enter the image name here that you want to be analyzed, this will store the image as a nxm matrix that will go through filters to do
# image processing
im = imread('pic2.jpg')
im = rgb2gray(im)
imshow(im)
plt.show()
imsave('s1.jpg', im)

app = ClarifaiApp('oar_ugwDyUJRO448FOMx6q3MlRT680u5vBjv6Pw7', '5ZA02j6KTN3JTbOQ1yJ6_WdSHDA1nSceNLFdWOup')
model = app.models.get('color', model_type='color')

img = ClImage(filename='s1.jpg')
info = model.predict([img])
#print info

# Print hex representation of the colors, the name of colors, and the percentage it covers on the image
for i in range(len(info['outputs'][0]['data']['colors'])):
    print str(info['outputs'][0]['data']['colors'][i]['raw_hex']) + "\t" + str(info['outputs'][0]['data']['colors'][i]['value'] * 100.0) + "%\t" +  str(info['outputs'][0]['data']['colors'][i]['w3c']['name'])

#############################################################################################
# im = scipy.misc.imread('search area 1a.jpg')
# im = im.astype('int32')
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag = numpy.hypot(dx, dy)  # magnitude
# mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
# imshow(mag)
# plt.show()

#i mg = cv2.imread('huge image1.JPG',0)

# Output dtype = cv2.CV_8U
# sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
# sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)

# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.show()
# 
# 
# img = cv2.imread('huge image1.jpg',0)
# 
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# 
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# 
# plt.show()
#############################################################################################
