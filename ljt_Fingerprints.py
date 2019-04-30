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
	gamma = 95 #default value
	#gamma = 70 
	for image in images:
		print(np.mean(image))
		print(np.std(image))
		print(type(image))
		enhanced = alpha + gamma * (image - np.mean(image)) / np.std(image)
		enhanced = np.clip(enhanced,0,255) #limita casos em que os valores sao menores que 0 ou maiores que 255
		print(type(enhanced))
		#plt.imshow(enhanced, cmap="gray")
		#plt.show()
		processedImages.append(enhanced)

	for image in processedImages:
		image = ndimage.median_filter(image, size=(5,5)) #tamanho do filtro de mediana
		

	subMatrixBlockSize = 5 #subdivide os blocos da imagem
	w_0 = 0.5 #valores padrao (slide 11)
	w_1 = 0.5

	for image in processedImages: #parece SUSPEITO! REVER!!!!
		###########Efetuando filtro de sobel##############
		sobelX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)#tamanho dos filtros de Sobel
		sobelY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
		##################################################
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
		alphaX = np.square(hipoteSobel)*np.cos(2*angleSobel) #computar os alphas
		alphaY = np.square(hipoteSobel)*np.sin(2*angleSobel)
		#plt.imshow(alphaX, cmap="gray")
		#plt.show()
		#plt.imshow(alphaY, cmap="gray")
		#plt.show()


		#blocking de Sir Benna (gradiente medio)
		img_alpha_x_block = [[np.sum(alphaX[index_y*subMatrixBlockSize: index_y*subMatrixBlockSize + subMatrixBlockSize, index_x*subMatrixBlockSize: index_x*subMatrixBlockSize + subMatrixBlockSize]) / subMatrixBlockSize**2
				for index_x in range(image.shape[0]//subMatrixBlockSize)]
				for index_y in range(image.shape[1]//subMatrixBlockSize)]
   
		img_alpha_y_block = [[np.sum(alphaY[index_y*subMatrixBlockSize: index_y*subMatrixBlockSize + subMatrixBlockSize, index_x*subMatrixBlockSize: index_x*subMatrixBlockSize + subMatrixBlockSize]) / subMatrixBlockSize**2
				for index_x in range(image.shape[0]//subMatrixBlockSize)]
				for index_y in range(image.shape[1]//subMatrixBlockSize)]

		#all gradient blocks
		#USAR NP.ARCTAN2 no slide 9!!!
		#for i in range (len(img_alpha_x_block)):
			#print(np.arctan2(img_alpha_x_block[i],img_alpha_y_block[i]) * 180 / np.pi)
		
		##Calcula o bloco central
		centralBlock_x = (image.shape[0]//subMatrixBlockSize)//2
		centralBlock_y = (image.shape[1]//subMatrixBlockSize)//2
		maxDistance =  math.sqrt(centralBlock_x*centralBlock_x + centralBlock_y*centralBlock_y)
		meanArray = []
		stdArray = []
		##percorre todos os blocos
		for index_x in range(image.shape[0]//subMatrixBlockSize):
			for index_y in range(image.shape[1]//subMatrixBlockSize):
				blockValueList = []
				for i in range(subMatrixBlockSize):
					for j in range(subMatrixBlockSize):
						blockValueList.append(image[index_x*i][index_y*j])
						#print(index_x,i,index_y,j)                    #mostra bloco atual
				blockMean = np.mean(blockValueList)
				blockStd = np.std(blockValueList)

				meanArray.append(blockMean)
				stdArray.append(blockStd)



		for index_x in range(image.shape[0]//subMatrixBlockSize):
			for index_y in range(image.shape[1]//subMatrixBlockSize):
				#print("============================")
				blockValueList = []
				####################################
				#POSSIVEL OTIMIZAR ESTE TRECHO:    #
				####################################
				for i in range(subMatrixBlockSize):
					for j in range(subMatrixBlockSize):
						blockValueList.append(image[index_x*i][index_y*j])
						#print(index_x,i,index_y,j)                    #mostra bloco atual
				blockMean = np.mean(blockValueList)
				blockStd = np.std(blockValueList)
				####################################


				blockMean = (blockMean - min(meanArray)) / (max(meanArray) - min(meanArray)) #normaliza entre 0 e 1
				blockStd = (blockStd - min(stdArray)) / (max(stdArray) - min(stdArray)) #normaliza entre 0 e 1
				#print(blockMean," ",blockStd)                             #mostra media e desvio padrao
				distanceToCenter = math.sqrt((centralBlock_x - index_x)**2 + (centralBlock_y - index_y)**2)
				w_2 = distanceToCenter / maxDistance
				v = w_0 * (1 - blockMean) + w_1 * blockStd + w_2
				#print(w_2," ",v)
				if v > 0.8:
					print ("Bloco ",index_x,",",index_y," = RI")
				else:
					print ("Bloco ",index_x,",",index_y," != RI")
					for i in range(subMatrixBlockSize):
						for j in range(subMatrixBlockSize):
							image[index_x+i][index_y+j] = 128

			plt.imshow(image, cmap="gray")
			plt.show()
			

			#
			#for index_x in range(img_alpha_x_block[i].shape[0])
			#	for index_y in range(img_alpha_y_block[i].shape[1])

			



		
		
		#plot_images()








if __name__ == "__main__":
	main()

			
