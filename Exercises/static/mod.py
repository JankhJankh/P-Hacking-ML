import scipy.misc
import os
from scipy import ndimage
import numpy as np

#This code will currently change the color of all car photos to be more red

#This gets the file names for every file in the current directory
for filename in os.listdir("."):
	#This makes it only choose image files
	if filename.endswith(".png"):
		#This reads the image to a 32x32x3 array called image
		image = ndimage.imread(filename)
		#If the current image is a car this code will iterate through every pixel, and add up to 50 to the red value of that pixel
		if filename.startswith("auto"):
			#Iterate through every X pixel
			for i in range(0,32):
				#Iterate through every Y pixel
				for j in range(0,32):
					#Check if adding 50 to the red pixel will put it above 255
					val = image[i][j][1] + 50
					#If the value is larger than the maximum, cap it at 255
					if(val + 50 > 255):
						val = 255
					#Add 50 to the red pixel (array index [1] in the 3rd dimension)
					image[i][j][1] =+ 50
			#Save the modified image
			scipy.misc.toimage(image, cmin=0.0, cmax=255).save(filename)