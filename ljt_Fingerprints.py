#!/usr/bin/python


###Autor: Leonardo Joji Takii
###Atividade pratica de biometria, ministrada pelo professor David Menotti, UFPR, 2019
###Identificacao de impressoes digitais
# Import the required modules
import cv2
import os
from PIL import Image, ImageDraw
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

poincareTolerance = 5
blockDimension = 15 #1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 100, 150 e 300 (divisores de 300)


signum = lambda x: -1 if x < 0 else 1
cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def get_angle(left, right):
	angle = left - right
	if abs(angle) > 180:
		angle = -1 * signum(angle) * (360 - abs(angle))
	return angle

#Baseado em:
#https://www.sciencedirect.com/science/article/pii/S1110866513000030
##NAO UTILIZADO NO MOMENTO
def smooth_gradient(orientation_blocks, blk_sz):
	orientation_blocks_smooth = np.zeros(orientation_blocks.shape)
	blk_no_y, blk_no_x = orientation_blocks.shape
	# Consistency level, filter of size (2*cons_lvl + 1) x (2*cons_lvl + 1)
	cons_lvl = 1
	for i in range(cons_lvl, blk_no_y-cons_lvl):
		for j in range(cons_lvl, blk_no_x-cons_lvl):
		  area_sin = area_cos = orientation_blocks[i-cons_lvl: i + cons_lvl, j-cons_lvl: j+cons_lvl]
											  
		  mean_angle_sin = np.sum(np.sin(2*area_sin))
		  mean_angle_cos = np.sum(np.cos(2*area_cos))
		  mean_angle = np.arctan2(mean_angle_sin, mean_angle_cos) / 2
		  orientation_blocks_smooth[i, j] = mean_angle

	return orientation_blocks_smooth

##Loop com tamanho de passo variavel, para ser utilizado em operacoes com blocos 
def drange(start, stop, step):
	while start < stop:
			yield start
			start += step

##NAO UTILIZADO NO MOMENTO
def plot_images(self,imgs,titles): #by bruno meyer
		fig = plt.figure(figsize=(12, 3))
		for i, a in enumerate(imgs):
			ax = fig.add_subplot(1, len(imgs), i + 1)
			ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
			ax.set_title(titles[i], fontsize=10)
			ax.set_xticks([])
			ax.set_yticks([])

def main():#PATH TESTADO: 'Rindex28/' 
	if path == 'Rindex28/' or path == 'Lindex101/' :
		image_paths = [os.path.join(path, f) for f in os.listdir(path) if  f.endswith('.raw')]
		# labels will contains the label that is assigned to the image
		images = []
		labels = []

		#print(image_paths)
		#print(os.listdir(path))
		print("Loading database: "+ path)
		for image_path in image_paths:	#LOOP PARA OBTER IMAGENS .raw
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
	for image in images:#APRIMORAMENTO DE IMAGEM
		#print(np.mean(image))
		#print(np.std(image))
		#print(type(image))
		enhanced = alpha + gamma * (image - np.mean(image)) / np.std(image)
		enhanced = np.clip(enhanced,0,255) #limita casos em que os valores sao menores que 0 ou maiores que 255
		#print(type(enhanced))
		#plt.imshow(enhanced, cmap="gray")
		#plt.show()
		processedImages.append(enhanced)

	for image in processedImages:#ELIMINA FALSOS GRADIENTES GERADOS POR RUIDO
		image = ndimage.median_filter(image, size=(5,5)) #tamanho do filtro de mediana
		

	subMatrixBlockSize = blockDimension #subdivide os blocos da imagem
	w_0 = 0.5 #valores padrao (slide 11)
	w_1 = 0.5

	for image in processedImages: #CALCULOS DOS GRADIENTES DOS BLOCOS E ZONAS DE INTERESSE
		
		"""###########Efetuando filtro de sobel##############
		sobelX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)#tamanho dos filtros de Sobel
		sobelY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
		##################################################
		hipoteSobel = np.sqrt(np.square(sobelX) + np.square(sobelY))#OBTEM A HIPOTENUSA FORMADA PELOS CATETOS DE SOBEL (letra grega Ro)

		angleSobel = np.arctan(np.divide(sobelY,sobelX,out=np.zeros_like(sobelY), where=sobelX!=0))#ARCO TANGENTE (ANGULO) BASEADO NOS VALORES DE SOBEL (letra grega Theta)
		
		
		#plt.imshow(sobelX, cmap="gray")
		#plt.show()
		#plt.imshow(sobelY, cmap="gray")
		#plt.show()
		#plt.imshow(hipoteSobel, cmap="gray")
		#plt.show()
		#plt.imshow(angleSobel, cmap="gray")
		#plt.show()
		alphaX = np.square(hipoteSobel)*np.cos(2*angleSobel) #COMPUTA OS ALPHAS 
		alphaY = np.square(hipoteSobel)*np.sin(2*angleSobel)
		#plt.imshow(alphaX, cmap="gray")
		#plt.show()
		#plt.imshow(alphaY, cmap="gray")
		#plt.show()
		
		#Blocking de Sir Benna  #CALCULA O GRADIENTE MEDIO DE CADA BLOCO DA IMAGEM
		img_alpha_x_block = [[np.sum(alphaX[index_y*subMatrixBlockSize: index_y*subMatrixBlockSize + subMatrixBlockSize, index_x*subMatrixBlockSize: index_x*subMatrixBlockSize + subMatrixBlockSize]) / subMatrixBlockSize**2
				for index_x in range(image.shape[0]//subMatrixBlockSize)]
				for index_y in range(image.shape[1]//subMatrixBlockSize)]
	
		img_alpha_y_block = [[np.sum(alphaY[index_y*subMatrixBlockSize: index_y*subMatrixBlockSize + subMatrixBlockSize, index_x*subMatrixBlockSize: index_x*subMatrixBlockSize + subMatrixBlockSize]) / subMatrixBlockSize**2
				for index_x in range(image.shape[0]//subMatrixBlockSize)]
				for index_y in range(image.shape[1]//subMatrixBlockSize)]
				
		#all gradient blocks
		#USAR NP.ARCTAN2 no slide 9!!!
		img_alpha_x_block = np.array(img_alpha_x_block) #transforma bloco em vetor
		img_alpha_y_block = np.array(img_alpha_y_block)
		#print (angles.shape[0])
		angles = np.zeros( (int(image.shape[0]/subMatrixBlockSize), int(image.shape[1]/subMatrixBlockSize)))
		#angles = np.zeros(len(img_alpha_x_block) * len(img_alpha_y_block) )
		for i in range (len(img_alpha_x_block)): #OBTEM O ANGULO DE CADA GRADIENTE MEDIO
			for j in range (len(img_alpha_y_block)):
				#angles[i,j] =  (np.arctan2(img_alpha_x_block[i],img_alpha_y_block[j]) * 0.5) 
				print( (np.arctan2(img_alpha_x_block[i],img_alpha_y_block[j]) * 0.5) )
				#print(np.arctan2(img_alpha_x_block[i],img_alpha_y_block[i]) * 180 / np.pi)
		
		
		"""
		##SIR BENNA METHOD
		blk_sz = subMatrixBlockSize

		dy = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
		dx = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

		img_alpha_x = dx*dx - dy*dy
		img_alpha_y = 2 * np.multiply(dx, dy)
	   
		img_alpha_x_block = [[np.sum(img_alpha_x[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
							for index_x in range(0, image.shape[0], blk_sz)]
					for index_y in range(0, image.shape[1], blk_sz)]
	   
		img_alpha_y_block = [[np.sum(img_alpha_y[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
							for index_x in range(0, image.shape[0], blk_sz)]
					for index_y in range(0, image.shape[1], blk_sz)]

		img_alpha_x_block = np.array(img_alpha_x_block)
		img_alpha_y_block = np.array(img_alpha_y_block)

		orientation_blocks = np.arctan2(img_alpha_y_block, img_alpha_x_block) / 2 #BLOCOS DE ORIENTACAO

		##SIR BENNA METHOD
		# Consistency level, filter of size (2*cons_lvl + 1) x (2*cons_lvl + 1)
		cons_lvl = 1  
		
		orientation_blocks_smooth = np.zeros(orientation_blocks.shape)
		blk_no_y, blk_no_x = orientation_blocks.shape
		
		for i in range(cons_lvl, blk_no_y-cons_lvl): #Obtem angulos suavizados
			for j in range(cons_lvl, blk_no_x-cons_lvl):
			  area_sin = area_cos = orientation_blocks[i-cons_lvl: i + cons_lvl, j-cons_lvl: j+cons_lvl]
												  
			  mean_angle_sin = np.sum(np.sin(2*area_sin))
			  mean_angle_cos = np.sum(np.cos(2*area_cos))
			  mean_angle = np.arctan2(mean_angle_sin, mean_angle_cos) / 2
			  orientation_blocks_smooth[i, j] = mean_angle
		
		orientation_blocks_smooth = np.array(orientation_blocks_smooth)
		#time.sleep(100)
		#show_orientation_map(img, ang)
		gradImg = image
		gradImg = np.where(gradImg > 200, 200, gradImg)
		gradImg = gradImg + 54
		tam = int(subMatrixBlockSize)
		onlyGrad = np.zeros_like(gradImg)
		np.where(onlyGrad == 0.0, 128, onlyGrad)
		#onlyGrad = onlyGrad.fill(255)
		angleIndex = 0
		halfBlock = subMatrixBlockSize//2
		imgs_grad = []
		for i in range(cons_lvl, blk_no_y-cons_lvl):#DESENHANDO OS GRADIENTES COMO RETAS
			for j in range(cons_lvl, blk_no_x-cons_lvl):

				x = int(np.cos(orientation_blocks_smooth[i,j])*(halfBlock-1))
				y = int(np.sin(orientation_blocks_smooth[i,j])*(halfBlock-1))
				#print(x," ",y)
						
				cv2.line(gradImg[i*tam:(i+1)*tam, j*tam:(j+1)*tam],(halfBlock+x, halfBlock-y), (halfBlock-x, halfBlock+y),(0,0,0),1)
				cv2.line(onlyGrad[i*tam:(i+1)*tam, j*tam:(j+1)*tam],(halfBlock+x, halfBlock-y), (halfBlock-x, halfBlock+y),(255,255,255),1)
				#print(angleIndex)
				#angleIndex += 1 
		imgs_grad.append(gradImg)
		#plt.imshow(cv2.resize(np.hstack((gradImg,onlyGrad)),None, fx=1.2, fy=1.2),cmap="gray")
		#plt.show()
		#cv2.imshow("mapa", cv2.resize(np.hstack((gradImg,onlyGrad)),None, fx=1.2, fy=1.2))
		#cv2.waitKey()
		###REGIOES DE INTERESSE
		##Calcula o bloco central
		centralBlock_x = (image.shape[0]//subMatrixBlockSize)//2
		centralBlock_y = (image.shape[1]//subMatrixBlockSize)//2
		maxDistance =  math.sqrt(centralBlock_x*centralBlock_x + centralBlock_y*centralBlock_y) #MAIOR DISTANCIA ATE O CENTRO
		meanArray = [] #ARMAZENA TODAS AS MEDIAS
		stdArray = [] #ARMAZENA TODOS OS DESV PADR
		##percorre todos os blocos
		roi_block = np.zeros(orientation_blocks.shape)
		for index_x in drange(0,image.shape[0],subMatrixBlockSize): #CALCULA MEDIA E STDEV ENTRE OS PIXELS DE UM BLOCO
			for index_y in drange(0,image.shape[1],subMatrixBlockSize):
				blockValueList = []
				for i in range(subMatrixBlockSize):
					for j in range(subMatrixBlockSize):
						blockValueList.append(image[index_x+i][index_y+j])
						#print(index_x,i,index_y,j)					#mostra bloco atual
				blockMean = np.mean(blockValueList) #Calcula media do bloco
				blockStd = np.std(blockValueList) #Calcula desvio padrao do bloco

				meanArray.append(blockMean)
				stdArray.append(blockStd)


		#blockIndex = 0
		for index_x in drange(0,image.shape[0],subMatrixBlockSize): #CALCULA MEDIA E STDEV ENTRE TODOS OS BLOCOS (MAL OTIMIZADO)
			for index_y in drange(0,image.shape[1],subMatrixBlockSize): #TRECHO PARA DETECTAR BLOCOS DE INTERESSE
				#print("============================")
				blockValueList = []
				####################################
				#POSSIVEL OTIMIZAR ESTE TRECHO:	#
				####################################
				for i in range(subMatrixBlockSize):
					for j in range(subMatrixBlockSize):
						blockValueList.append(image[index_x+i][index_y+j])
						#print(index_x,i,index_y,j)					#mostra bloco atual
				blockMean = np.mean(blockValueList)
				blockStd = np.std(blockValueList)
				####################################


				blockMean = (blockMean - min(meanArray)) / (max(meanArray) - min(meanArray)) #normaliza entre 0 e 1
				blockStd = (blockStd - min(stdArray)) / (max(stdArray) - min(stdArray)) #normaliza entre 0 e 1
				#print(blockMean," ",blockStd)							 #mostra media e desvio padrao
				distanceToCenter = math.sqrt((centralBlock_x - index_x//subMatrixBlockSize)**2 + (centralBlock_y - index_y//subMatrixBlockSize)**2)
				w_2 = 1 - (distanceToCenter / maxDistance)
				v = w_0 * (1 - blockMean) + w_1 * blockStd + w_2
				#print(w_2," ",v)
				if v < 0.8:
					orientation_blocks[index_x//subMatrixBlockSize,index_y//subMatrixBlockSize] = 0.0
					orientation_blocks_smooth[index_x//subMatrixBlockSize,index_y//subMatrixBlockSize] = 0.0
					roi_block[index_x//subMatrixBlockSize,index_y//subMatrixBlockSize] = 1
					#print ("Bloco ",index_x,",",index_y," != RI")
					for i in range(subMatrixBlockSize):
						for j in range(subMatrixBlockSize):
							
							image[index_x+i][index_y+j] = 0
							onlyGrad[index_x+i][index_y+j] = 0
				#blockIndex = blockIndex + 1
		#plt.imshow(cv2.resize(np.hstack((gradImg,onlyGrad,image)),None, fx=1.2, fy=1.2),cmap="gray")
		#plt.show()
		#plt.imshow(image, cmap="gray")
		#plt.show()
		
		
		W = subMatrixBlockSize
		#result = calculate_singularities(im, angles, int(args.tolerance[0]), W)


		tolerance = poincareTolerance #Temporary	
		featureImage = Image.fromarray(image)
		(x, y) = featureImage.size
		result = featureImage.convert("RGBA")
		featureArray = np.array(featureImage)
		#result = image#temporary
		draw = ImageDraw.Draw(result)

		#orientation_blocks = orientation_blocks_smooth

		colors = {"loop" : (150, 0, 0, 0), "delta" : (0, 150, 0, 0), "whorl": (0, 0, 150, 0)}
		poincare = np.zeros(orientation_blocks.shape)
		##Deteccao de Cores e Deltas
		##Baseado na implementacao do Poincare de https://github.com/rtshadow/biometrics/blob/master/poincare.py
		
		def draw_singular_points(image, singular_pts, poincare, blk_sz, thicc=2):
		   '''
			  image - Image array.
			  poincare - Poincare index matrix of each block.
			  blk_sz - block size of the orientation field.
			  tolerance - Angle tolerance in degrees.
			  thicc - Thickness of the lines to plot.
		   '''
		   # add color channels to ndarray
		   if(len(image.shape) == 2):
			  image_color = np.stack((image,)*3, axis=-1)
		   else:
			  image_color = image

		   # cores, deltas, whorls = singular_pts/blk_sz
		   divide_tuple_by_blk_sz = lambda point_blk: (point_blk[0]/blk_sz, point_blk[1]/blk_sz)
		   cores, deltas, whorls = [map(divide_tuple_by_blk_sz, singular_pts_typed)
									for singular_pts_typed in singular_pts]

		   for i, j in map(lambda x: (int(x[0]), int(x[1])), cores):
			  color = matplotlib.cm.hot(abs(poincare[i, j]/360))
			  color = (color[0]*255, color[1]*255, color[2]*255)
			  cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
						  int(blk_sz/2), color, thicc)

		   for i, j in map(lambda x: (int(x[0]), int(x[1])), deltas):
			  cv2.rectangle(image_color, (j*blk_sz, i*blk_sz),
							((j+1)*blk_sz, (i+1)*blk_sz), (0, 125, 255), thicc)

		   for i, j in map(lambda x: (int(x[0]), int(x[1])), whorls):
			  cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
						  int(blk_sz/2), (0, 200, 200), thicc)
		   return image_color
		
		for i in range(1, len(orientation_blocks_smooth[0]) - 1): 
			for j in range(1, len(orientation_blocks_smooth[1]) - 1):
				#print(orientation_blocks_smooth)
				if roi_block[i][j] != 0 or roi_block[i+1][j] != 0 or roi_block[i-1][j] != 0 or roi_block[i][j+1] != 0 or roi_block[i][j-1] != 0:
					continue
				deg_angles = [math.degrees(orientation_blocks_smooth[i - k][j - l]) % 180 for k, l in cells]
				index = 0
				#print(deg_angles)
				for k in range(0, 8):
					if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
						deg_angles[k + 1] += 180

					index += get_angle(deg_angles[k], deg_angles[k + 1])
				poincare[i,j] = index
				singularity = "none"
				if (np.isclose(poincare[i, j], 180, 0, tolerance)):
					singularity = "loop"
					draw.ellipse([(j * W, i * W), ((j + 1) * W, (i + 1) * W)],fill=(64,64,64,0), outline = colors[singularity])
					cv2.circle(image, (int((j+0.5)*W), int((i+0.5) * W)),
						  int(W/2), (128,0,0), 2)
					#cores.append((i, j))
				if (np.isclose(poincare[i, j], -180, 0, tolerance)):
					singularity = "delta"
					draw.rectangle([(j * W, i * W), ((j + 1) * W, (i + 1) * W)],fill=(255,255,255,0), outline = colors[singularity])
					cv2.rectangle(image, (j*W, i*W), ((j+1)*W, (i+1)*W), (0, 125, 255), 2)
					#deltas.append((i, j))
				if (np.isclose(poincare[i, j], 360, 0, tolerance)):
					singularity = "whorl"
					draw.ellipse([(j * W, i * W), ((j + 1) * W, (i + 1) * W)],fill=(128,128,128,0), outline = colors[singularity])
					#whorls.append((i, j))
				"""
				if 180 - tolerance <= index and index <= 180 + tolerance:
					singularity = "loop"
					draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)],fill=(64,64,64,0), outline = colors[singularity])
					print(index),
				elif -180 - tolerance <= index and index <= -180 + tolerance:
					singularity = "delta"
					draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)],fill=(255,255,255,0), outline = colors[singularity])
					print(index),
				"""
				#if 360 - tolerance <= index and index <= 360 + tolerance:
				#	singularity = "whorl"
				#if 180 == index :
				#	singularity = "loop"
					#draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)], outline = colors[singularity])
				#if -180 == index:
				#	singularity = "delta"
					#draw.rect([(i * W, j * W), ((i + 1) * W, (j + 1) * W)], outline = colors[singularity])
				#if 360 == index:
					#singularity = "whorl"			

				if singularity != "none":
					print(singularity,i,j)					
					#orientation_blocks_smooth[i][j] = 0.0
					#featureArray[x*subMatrixBlockSize+i][y*subMatrixBlockSize+j] = 255

					#draw.ellipse([(i * W, j * W), ((i + 1) * W, (j + 1) * W)],fill=(255,255,255,0), outline = colors[singularity])

		del draw
		#featureImage = Image.alpha_composite(featureImage, result)
		#print("draw")
		#featureImage += draw
		result = result.convert("L")
		result = np.array(result)
		print("")
		print("=========================")
		#plt.imshow(result,cmap="pink")
		plt.imshow(cv2.resize(np.hstack((gradImg,image,onlyGrad,result)),None, fx=1.2, fy=1.2))
		plt.show()
		#result.show()
	
	

	"""
	def poincare_index_at(i, j, orientation_blocks, tolerance):
		deg_angles = [math.degrees(orientation_blocks[i - k][j - l]) % 180 for k, l in cells]
		index = 0
		for k in range(0, 8):
			if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
				deg_angles[k + 1] += 180
			index += get_angle(deg_angles[k], deg_angles[k + 1])

		if 180 - tolerance <= index and index <= 180 + tolerance:
			singularity = "loop"
		if -180 - tolerance <= index and index <= -180 + tolerance:
			singularity = "delta"
		if 360 - tolerance <= index and index <= 360 + tolerance:
			singularity = "whorl"
		singularity = "none"
	"""

		
		

if __name__ == "__main__":
	main()

			
