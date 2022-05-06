import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt
from numpy.linalg import inv
import numpy as np


def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if(line!="\n"):    
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

    return data



calib1 = read_calib_file("C:/Users/sebac/Desktop/uni/9.tesi magistrale/Codice/Kitti/bho/testing/calib_cam_to_cam/000000.txt")


#read image and show
imgL = cv2.imread('C:/Users/sebac/Desktop/uni/9.tesi magistrale/Codice/Kitti/bho/testing/image_2/000000_10.png')
imgR = cv2.imread('C:/Users/sebac/Desktop/uni/9.tesi magistrale/Codice/Kitti/bho/testing/image_3/000000_10.png')
cv2.imshow("frame left", imgL)
# cv2.imshow("frame right",imgR)

#downsampling
imgL = downsample_image(imgL,1)
imgR = downsample_image(imgR,1)

# cv2.imshow("frame left half",imgLhalf)


#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#=========================================================
# Create Disparity map from Stereo Vision
#=========================================================

# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error. 
block_size = 5
min_disp = -1
max_disp = 31
num_disp = max_disp - min_disp # Needs to be divisible by 16

# Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = block_size,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 2,
	disp12MaxDiff = 2,
	P1 = 8 * 3 * block_size**2,#8*img_channels*block_size**2,
	P2 = 32 * 3 * block_size**2) #32*img_channels*block_size**2)


#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=win_size)

# Compute disparity map
disparity_map = stereo.compute(imgLgray, imgRgray)
# disparity_map = stereo.compute(imgL, imgR)

# Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()


#stereo rectifity variables

K_02 = np.reshape(calib1['K_02'], (3, 3))
D_02 = calib1['D_02']
K_03 = np.reshape(calib1['K_03'], (3, 3))
D_03 = calib1['D_03']
print(imgLgray.shape[::-1])
R_03 = np.reshape(calib1['R_03'], (3, 3))
R_02 = np.reshape(calib1['R_02'], (3, 3))

R_02inv = inv(R_02)

rotation = np.matmul( R_02inv, R_03)
T_02 = (calib1['T_02'])
T_02inv = np.reshape(T_02, (1, 3))
T_03 = np.reshape(calib1['T_03'], (1, 3))
translation  = np.reshape(np.add( T_02inv,T_03),(3, 1))

#convert in np matrix
K_02 = np.asmatrix(K_02)
D_02 = np.asmatrix(D_02)
K_03 = np.asmatrix(K_03)
D_03 = np.asmatrix(D_03)
R_03 = np.asmatrix(R_03)
R_02 = np.asmatrix(R_02)
rotationb = np.asmatrix(rotation)
T_02 = np.asmatrix(T_02)
translationb  = np.asmatrix(translation)

# output matrices from stereoRectify init
R1 = np.zeros(shape=(3, 3))
R2 = np.zeros(shape=(3, 3))
P1 = np.zeros(shape=(3, 4))
P2 = np.zeros(shape=(3, 4))
Q = np.zeros(shape=(4, 4))

rectifyScale= 1
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K_02, D_02, K_03, D_03, imgLgray.shape[::-1], rotation, translation ,rectifyScale, (0,0),flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
print(Q)

#=========================================================
# Generate Point Cloud from Disparity Map
#=========================================================

# Get new downsampled width and height 
h,w = imgR.shape[:2]

# Convert disparity map to float32 and divide by 16 as show in the documentation
print(disparity_map.dtype)
disparity_map = np.float32(np.divide(disparity_map, 16.0))
print(disparity_map.dtype)

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
# Get color of the reprojected points
colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]


# Function to create point cloud file
def create_point_cloud_file(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


output_file = 'pointCloud.ply'


create_point_cloud_file(output_points, output_colors, output_file)
print("end")