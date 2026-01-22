# This is CNN convolution example

data=[
    [1,2,3,4,5],
    [0,1,2,3,4],
    [2,1,0,1,2],
    [1,2,1,2,1],
    [0,1,0,1,0]
]

for i in range(len(data)):
	for j in range(len(data[0])):
		print(data[i][j], end=' ')
	print()

kernel=[
	[-1,-1,-1],
	[-1,8,-1],
	[-1,-1,-1]
]

# process upper left 3x3 pixels
# result=0
# for i in range(3):
# 	for j in range(3):
# 		result+=data[i][j]*kernel[i][j]
# 		print(f"data[{i}][{j}] * kernel[{i}][{j}] = {data[i][j]} * {kernel[i][j]} = {data[i][j]*kernel[i][j]}")
  
# print("Convolution result at (0,0):", result)

# process entire image with valid padding
output_height=len(data)-len(kernel)+1
output_width=len(data[0])-len(kernel[0])+1
output=[[0 for _ in range(output_width)] for _ in range(output_height)]
for i in range(output_height):
	for j in range(output_width):
		result=0
		for ki in range(len(kernel)):
			for kj in range(len(kernel[0])):
				result+=data[i+ki][j+kj]*kernel[ki][kj]
		output[i][j]=result
  
print("Convolution output:")
for row in output:
	for val in row:
		print(val, end=' ')
	print()
 
# max pooling with 2x2

data=[
	[1,3,2,4],
	[5,6,1,2],
	[7,8,3,1],
	[2,1,4,5]
]

pool_size=2
output_height=len(data)//pool_size
output_width=len(data[0])//pool_size
pooled_output=[[0 for _ in range(output_width)] for _ in range(output_height)]
for i in range(output_height):
	for j in range(output_width):
		max_val=float('-inf')
		for pi in range(pool_size):
			for pj in range(pool_size):
				if data[i*pool_size+pi][j*pool_size+pj]>max_val:
					max_val=data[i*pool_size+pi][j*pool_size+pj]
		pooled_output[i][j]=max_val
  
# print pooled output
print("Max Pooling output:")
for row in pooled_output:
	for val in row:
		print(val, end=' ')
	print()