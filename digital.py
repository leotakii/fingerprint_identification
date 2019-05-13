import cv2
import numpy as np
import sys, os
from PIL import Image

def equaliza(img):
    return cv2.equalizeHist(img)
    
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)][:14]
    images = []
    labels = []
    for image_path in [x for x in image_paths if ('.raw')]:
        img = np.fromfile(image_path, dtype=np.uint8)
        image = img.reshape((300,300))
        #nbr = int(image_path.split(os.sep)[-1].replace('f','').replace('.raw','').split('R')[0])
        images.append(image)
        #labels.append(nbr)
    return np.array(images), np.array(labels)


def preprocessa(imgs, alpha=150, y=95):
    new_imgs = []
    for img in imgs:
        #imga = cv2.medianBlur(img,5)
        imga = img
        media = np.mean(imga)
        var = np.var(imga)
        funcao = lambda el: alpha + y * ((el - media)/var)
        operacao = np.vectorize(funcao)
        res = operacao(imga.astype(np.float))
        res = np.where(res < 0, 0, res)
        res = np.where(res > 255, 255, res)
        res = (((res-np.min(res))/(np.max(res)-np.min(res)))*255).astype('uint8')
        res = cv2.medianBlur(res,5)
#        k = np.ones((3,3))
#        res = cv2.dilate(res, k, iterations=1)
#        res = np.where(res > 180, 255, res)
        new_imgs.append(res)
        #new_imgs.append(cv2.medianBlur(res,3))
        #new_imgs.append(res)
        
    return np.array(new_imgs)

def angle(x,y):
    if (x >= 0):
        return np.arctan(y/x)
    else:
        if (y >= 0):
            return (np.arctan(y/x) + np.pi)
        else:
            return (np.arctan(y/x) - np.pi)

def normaliza(img):
    return (255*((img - np.min(img))/(np.max(img)-np.min(img)))).astype('uint8')

def mapa(imgs, b_size):
    import math
    ang_imgs = []
    for img in imgs:
        tam = int(img.shape[0]/b_size)
        gx = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        ax = np.zeros((int(img.shape[0]/b_size), int(img.shape[1]/b_size)))
        ay = np.zeros((int(img.shape[0]/b_size), int(img.shape[1]/b_size)))
        ang = np.zeros((int(img.shape[0]/b_size), int(img.shape[1]/b_size)))
        stride = b_size
        for i in range(tam):
            for j in range(tam):
                Gx = gx[stride*i:stride*(i+1), stride*j:stride*(j+1)]
                Gy = gy[stride*i:stride*(i+1), stride*j:stride*(j+1)]
                x = np.sum((Gx*Gx)-(Gy*Gy))/tam*tam
                y = np.sum(2*np.multiply(Gx,Gy))/tam*tam
                res = angle(x,y)/2.0
                ang[i,j] = 0 if np.isnan(res) else res

        ang_imgs.append(ang)
    return np.array(ang_imgs)


def desenha_grad(imgs, angs, b_size, tchk=2):
    imgs_grad = []
    for img, ang in zip(imgs, angs):
#        show_orientation_map(img, ang)
        img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img2 = np.where(img2>200, 200, img2)
        img2 = img2+54
        tam = int(b_size)
        img3 = np.zeros_like(img2)
        for i in range(ang.shape[0]):
            for j in range(ang.shape[1]):
                metade = int(b_size/2)
                x = int(np.cos(ang[i,j])*(metade-tchk))
                y = int(np.sin(ang[i,j])*(metade-tchk))
                #print(x," ",y) 
                #x = int(np.cos(ang[i,j]*np.pi/180)*(metade-tchk))
                #y = int(np.sin(ang[i,j]*np.pi/180)*(metade-tchk))
                cv2.line(img2[i*tam:(i+1)*tam, j*tam:(j+1)*tam],(metade+x, metade-y), (metade-x, metade+y),(0,0,255),tchk)
                cv2.line(img3[i*tam:(i+1)*tam, j*tam:(j+1)*tam],(metade+x, metade-y), (metade-x, metade+y),(0,0,255),tchk)
        imgs_grad.append(img2)
        cv2.imshow("mapa", cv2.resize(np.hstack((img2,img3)),None, fx=1.2, fy=1.2))
        cv2.waitKey()
                
def detecta_ri(imgs, b_size):
    for img in imgs:
        tam = int(img.shape[0]/b_size)
        stride = b_size
        media = np.zeros((int(img.shape[0]/b_size), int(img.shape[1]/b_size)))
        std = np.zeros((int(img.shape[0]/b_size), int(img.shape[1]/b_size)))
        for i in range(tam):
            for j in range(tam):
                area = img[stride*i:stride*(i+1), stride*j:stride*(j+1)]
                media[i,j] = np.mean(area)
                std[i,j] = np.std(area)
        media = (media-np.min(media))/(np.max(media) - np.min(media))
        std = (std-np.min(std))/(np.max(std) - np.min(std))
        for i in range(tam):
            for j in range(tam):
                w0 = 0.5
                w1 = 0.5
                w2 = 1 - (np.sqrt(np.power(i-(tam/2),2)+np.power(j-(tam/2),2))/tam)
                if (w0*(1-media[i,j])+w1*std[i,j]+w2 < 0.8):
                    img[stride*i:stride*(i+1), stride*j:stride*(j+1)] = 125
        cv2.imshow("imagem", img)
        cv2.waitKey(240)

path = sys.argv[1]
images, labels = get_images_and_labels (path)
print("processa")
images_p = images
images_p = preprocessa(images)
block_size = 11
ang_images = mapa(images_p, block_size)
desenha_grad(images_p, ang_images, block_size)
#detecta_ri(images_p, block_size)
'''
for img,img2 in zip(images,images_p):
    cv2.imshow("dig",np.hstack((img,img2)))
    cv2.waitKey(350)
'''

