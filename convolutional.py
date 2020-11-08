import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage, signal

# Opening the image and resizing it
unProcessedImage = Image.open('image.png')
unProcessedImage.thumbnail((50, 50))

# Creating a numpy matrix with pixel values (only black and white)
imageMatrix = np.asarray(unProcessedImage.convert('L'), dtype=np.float32)

# PROCESSING
h_kernel = np.array([ [-1, -1, -1], 
                    [1, 1, 1], 
                    [0, 0, 0]])

v_kernel = np.array([ [1, -1, 0], 
                    [1, -1, 0], 
                    [1, -1, 0]])

g_kernel = np.array([ [1, 2, 1], 
                    [2, 4, 2], 
                    [1, 2, 1]])

processedImageMatrix = ndimage.convolve(imageMatrix, v_kernel)

# Plotting the pixel values on a graph
plt.imshow(processedImageMatrix, cmap = 'gray')
plt.axis('off')
plt.show()

