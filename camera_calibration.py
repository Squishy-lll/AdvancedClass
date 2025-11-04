import numpy as np
import cv2 as cv
import glob

# measured side length of a chessboard square
square_size = 0.1

# input the amount of rows and columns that corners are visible
row_size = 7
column_size = 6

# create zero array of 3D points for each corner of chessboard (32bits more reliable for opencv)
real_board_points = np.zeros((row_size*column_size,3), np.float32)

# assign coords to corners, top left corner is origin, x to the right is + and y down is +, z axis = 0
real_board_points[:,:2] = np.mgrid[0:row_size,0:column_size].T.reshape(-1,2) * square_size


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria: stop if iterations = 30 or pixel accuracy is less than epsilon value(0.001)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# window size of subpixel estimate for corner coords
pixel_search_area = (11,11)

# deadzone for subpixel estimate for corner coords (set to None)
deadzone = (-1,-1)

path = r'c:\Users\camel\Desktop\python_work\AdvancedClass\chessboard_images'
images = glob.glob("path\*.jpg")

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners_found, corner_points = cv.findChessboardCorners(image, (row_size,column_size), None)

    if corners_found == True:

        objpoints.append(real_board_points)

        #looks at the color gradient and makes the corner point estimate more accurate
        refined_corner_points =  cv.cornerSubPix(gray,corner_points, pixel_search_area, deadzone, criteria)
        imgpoints.append(refined_corner_points)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (row_size,column_size), refined_corner_points, corners_found)
        cv.imshow('img', img)
        cv.waitKey(500) # pause for half a second to examine process went well

cv.destroyAllWindows()

error, calibration_matrix, distortion_coefficents, orientation, position = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

