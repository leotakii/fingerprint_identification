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

def plot_images(self,imgs,titles): #by bruno meyer
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate(imgs):
            ax = fig.add_subplot(1, len(imgs), i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

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

	alpha = 150
	gamma = 95 #default
	#gamma = 70 
	for image in images:
		print(np.mean(image))
		print(np.std(image))
		print(type(image))
		enhanced = alpha + gamma * (image - np.mean(image)) / np.std(image)
		print(type(enhanced))
		#plt.imshow(enhanced, cmap="gray")
		#plt.show()
		processedImages.append(enhanced)






	for image in processedImages:
		image = ndimage.median_filter(image, size=(5,5)) #tamanho do filtro de mediana
		

	for image in processedImages: #parece SUSPEITO! REVER!!!!

		sobelX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)#tamanho dos filtros de Sobel
		sobelY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

		hipoteSobel = np.sqrt(np.square(sobelX) + np.square(sobelY))

		##trata divisoes por 0. REVER!!
		angleSobel = np.arctan(np.divide(sobelY,sobelX,out=np.zeros_like(sobelY), where=sobelX!=0))
		#plt.imshow(sobelX, cmap="gray")
		#plt.show()
		#plt.imshow(sobelY, cmap="gray")
		#plt.show()
		#plt.imshow(hipoteSobel, cmap="gray")
		#plt.show()
		#plt.imshow(angleSobel, cmap="gray")
		#plt.show()

		alphaX = np.square(hipoteSobel)*np.cos(2*angleSobel)
		alphaY = np.square(hipoteSobel)*np.sin(2*angleSobel)
		plt.imshow(alphaX, cmap="gray")
		plt.show()
		plt.imshow(alphaY, cmap="gray")
		plt.show()

		#plot_images()




#USAR NP.ARCTAN2 no slide 9!!!



if __name__ == "__main__":
	main()
