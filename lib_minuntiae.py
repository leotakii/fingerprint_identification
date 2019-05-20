## Versao modificada de Bruno Serbena
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm
import random
import colorsys
import cv2
import math
import sys
import time

def minutiae(image_spook, roi_blks, blk_sz, radius=16):
   # 0, 1, 3, 4 neighbors
   minutiae_list = [[],[],[],[],[]]
   minutiae_type = np.full(image_spook.shape, -1)

   for i in range(radius, image_spook.shape[0] - radius):
      for j in range(radius, image_spook.shape[1] - radius):
         # not in RoI or is background, skip it
         if(roi_blks[i//blk_sz, j//blk_sz] == 1 or image_spook[i,j] == 0):
            continue
         eight_nei = image_spook[i-1 : i+1+1 , j -1: j +1+1]
         eight_nei_no = np.sum(eight_nei) - 1
         minutiae_type[i,j] = eight_nei_no

   def clear_noise(image_spook, i, j, radius):

      block = image_spook[i-radius: i + radius, j-radius: j + radius]
      
      # (lambda x: 0 if(x == 1 or x == 3))(block)
      changed = False

      if minutiae_type[i, j] == 2 or minutiae_type[i, j] == 3 or minutiae_type[i, j] == 4:
         for l in range(i-radius, i + radius):
            for m in range(j-radius, j + radius):
               # ignora o indice atual
               if(l == i and j == m):
                  continue
               if(minutiae_type[l, m] == 3
                  or minutiae_type[l, m] == 4
                  or minutiae_type[l, m] == 2):
                  changed = True
                  minutiae_type[l, m] = -1
         if(changed):
            minutiae_type[i, j] = -1
      return

   for i in range(blk_sz, image_spook.shape[0] - blk_sz):
      for j in range(blk_sz, image_spook.shape[1] - blk_sz):
         block_x = i//blk_sz
         block_y = j//blk_sz
         if (block_y + 1 >= roi_blks.shape[1] or block_x + 1 >= roi_blks.shape[0] or block_y - 1 <= 0 or block_x - 1 <= 0 ):
            continue
         if ((roi_blks[block_x+1][block_y-1] != 0) or (roi_blks[block_x+1][block_y+1] != 0) or (roi_blks[block_x-1][block_y+1] != 0) or (roi_blks[block_x-1][block_y-1] != 0)):
           continue
		 
         #if((roi_blks[i + blk_sz, j - blk_sz] == 1) or (roi_blks[i + blk_sz, j + blk_sz] == 1) or (roi_blks[i - blk_sz, j + blk_sz] == 1) or (roi_blks[i - blk_sz, j - blk_sz] == 1)):
         #   continue
         #print(i)
         if(roi_blks[i//blk_sz, j//blk_sz] == 1 or minutiae_type[i, j] == -1):
            continue
         if(minutiae_type[i,j] == 0):
            # 'isolated'
            minutiae_list[0].append((i,j))

         elif(minutiae_type[i,j] == 1):
            # 'ending'
            #clear_noise(minutiae_type, i, j, radius)

            # if still minutiae after clearing
            if(minutiae_type[i, j] == 1):
               minutiae_list[1].append((i, j))

         elif(minutiae_type[i,j] == 2):
            # 'edgepoint'
            #pass
              clear_noise(minutiae_type, i, j, radius)
              minutiae_list[2].append((i,j))

         elif(minutiae_type[i,j] == 3):
            # 'bifurcation'
            #clear_noise(minutiae_type, i, j, radius)

            # if still minutiae after clearing
            if(minutiae_type[i, j] == 3):
               #pass
               minutiae_list[3].append((i,j))

         elif(minutiae_type[i,j] == 4):#GREEN
            # 'crossing'
            #pass
            #clear_noise(minutiae_type, i, j, radius)
            minutiae_list[4].append((i,j))
               

   return minutiae_list


def minutiae_draw(image_spook, minutiae_list, size=1, thicc=2):
   image_spook = image_spook*255

   # add color channels to ndarray
   if(len(image_spook.shape) == 2):
      image_color = np.stack((image_spook,)*3, axis=-1)
   else:
      image_color = image_spook

   resize_mult = 4
   image_color = cv2.resize(
       image_color, (resize_mult*image_color.shape[0], resize_mult*image_color.shape[1]), interpolation=cv2.INTER_NEAREST)

   #random.seed(133)
   random.seed(133)
   # yellow isolated point
   # purple endpoint
   #
   # red/pink bifurcation
   # green crossing

   for i in range(len(minutiae_list)):
      minutiae_list_typed = minutiae_list[i]
      # get next 'random' color in sequence
      #h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
      h, s, l = random.random(), 1 , 0.5
      color = r, g, b = [int(255*(i)) for i in colorsys.hls_to_rgb(h, l, s)]
      #print(minutiae_list_typed,color)
      for minu in minutiae_list_typed:
         i, j = minu
         cv2.circle(image_color, (int(j*resize_mult+resize_mult/2), int(i*resize_mult+resize_mult/2)),
                    size, color, thicc)
      i += 1
   return image_color
