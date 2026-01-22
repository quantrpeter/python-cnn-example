# This is CNN convolution example, read image and do convolution and pooling

import numpy as np
 
# read All Q Logos_colorful.png into variable data
def read_image_into_two_dimension_array(file_path):
	from PIL import Image
	import numpy as np

	# Open the image file
	img = Image.open(file_path)

	# Convert image to RGB (in case it's RGBA or grayscale)
	img = img.convert('RGB')

	# Convert image data to a 2D array (height x width x channels)
	data = np.array(img)

	return data

# Example usage:
image_path = 'All Q Logos_colorful.png'  # Replace with your image file path
data = read_image_into_two_dimension_array(image_path)

print(data.shape)  # Should print (height, width, channels)

# convolution operation
def convolution2d(image, kernel, stride=1, padding=0):
	# Add padding to the image
	if padding > 0:
		image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

	image_height, image_width, num_channels = image.shape
	kernel_height, kernel_width = kernel.shape

	# Calculate output dimensions
	out_height = (image_height - kernel_height) // stride + 1
	out_width = (image_width - kernel_width) // stride + 1

	# Initialize output feature map
	output = np.zeros((out_height, out_width, num_channels))

	# Perform convolution
	for c in range(num_channels):
		for i in range(0, out_height):
			for j in range(0, out_width):
				region = image[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width, c]
				output[i, j, c] = np.sum(region * kernel)

	return output

kernel=[
	[-1,-1,-1],
	[-1,8,-1],
	[-1,-1,-1]
]

# convolution2d(data, np.array(kernel), stride=1, padding=1)
# print(data.shape)
# create GUI show the image
def show_image(data):
	from PIL import Image
	import numpy as np

	# Convert the 2D array back to an image
	img = Image.fromarray(np.uint8(data))

	# Show the image
	img.show()
 
show_image(data)