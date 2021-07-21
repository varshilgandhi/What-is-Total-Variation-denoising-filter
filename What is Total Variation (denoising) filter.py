# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:04:19 2021

@author: abc
"""

"""

Total Variation denoising filter

"""

import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt

#read our image and convert it into floating point value
img = img_as_float(io.imread("BSE_25sigma_noisy.jpg", as_gray=True))

#denoise image using Total Variation denoising filter
denoise_img = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)

#plot histogram
plt.hist(denoise_img.flat, bins=100, range=(0,1))

#show our images
cv2.imshow("Original image", img)
cv2.imshow("Denoise image using Total variation filter", denoise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



