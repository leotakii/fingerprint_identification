#!/usr/bin/python


###Autor: Leonardo Joji Takii
###Atividade pratica de biometria, ministrada pelo professor David Menotti, UFPR, 2019
###Identificacao de impressoes digitais
# Import the required modules
import cv2
import os
from PIL import Image
import sys
import getopt
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import math
import random
from scipy import ndimage


import rawpy
#import imageio


# avoiding annoying warnings
import warnings
import matplotlib.cbook


# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

#Decide qual base sera utilizada

#path = 'Lindex101/'
path = 'Rindex28/'
#path = 'Rindex28-type/'

def main():
	if path == 'Rindex28/' or path == 'Lindex101/' :
		image_paths = [os.path.join(path, f) for f in os.listdir(path) if  f.endswith('.raw')]
		# labels will contains the label that is assigned to the image
		images = []
		labels = []

		#print(image_paths)
		#print(os.listdir(path))
		print("Loading database: "+ path)
		for image_path in image_paths:
			#print(image_path)

			A = np.fromfile(image_path, dtype='int8', sep="")

			#print(A.shape)
			A = A.reshape([300, -1])

			A = Image.fromarray(A)

			images.append(A)



			#plt.imshow(A, cmap="gray")
			#plt.show()

	else:
		return

	processedImages = []
	#enhancing image
	for image in images:
		print(np.mean(image))
		print(np.std(image))
		print(type(image))
		enhanced = 150 + 95 * (image - np.mean(image)) / np.std(image)
		print(type(enhanced))
		#plt.imshow(enhanced, cmap="gray")
		#plt.show()
		processedImages.append(enhanced)




	for image in processedImages:
		image = ndimage.median_filter(image, size=(5,5))
		plt.imshow(image, cmap="gray")
		plt.show()





if __name__ == "__main__":
	main()
