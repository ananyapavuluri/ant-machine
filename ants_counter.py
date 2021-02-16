import cv2
import matplotlib.pyplot
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Read BGR image
img = cv2.imread('ants_no_blob.jpeg')
original = img.copy()
background = cv2.imread('ants_background.jpeg')
# Converting color from BGR to RGB
# no_bk = cv2.subtract(img, background)
# cv2.imwrite('sub.jpg', no_bk)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_bk = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
gray_bk = cv2.cvtColor(rgb_bk, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
blur_bk = cv2.GaussianBlur(gray_bk, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh_bk = cv2.threshold(blur_bk, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

no_bk = cv2.subtract(thresh, thresh_bk)
cv2.imwrite('sub_threshold.jpg', no_bk)


# # Declaring some variables
# MIN,LOW,HUE = 0, 0, 0
# MAX,HIGH,SAT = 1, 1, 1
# VAL = 2
# # Tuple for storing HSV range
# HSV_Range = ((0, 0, 0), (179, 255, 255))
# # List for storing HSV values
# HSV_Values = [[20, 170, 170], [179, 255, 255]]
# # Storing HSV values in a scalar array
# lower_HSV = np.array([HSV_Values[LOW][HUE],HSV_Values[LOW][SAT],HSV_Values[LOW][VAL]])
# upper_HSV = np.array([HSV_Values[HIGH][HUE],HSV_Values[HIGH][SAT],HSV_Values[HIGH][VAL]])

# # Converting BRG image to HSV image
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
# processed_img = cv2.inRange(hsv_img, lower_HSV, upper_HSV)
# cv2.imwrite('thresh_img.jpg', processed_img)

# Getting rid of grains and noise
kernel_4 = np.ones((4,4),np.uint8)
kernel_2 = np.ones((2,2),np.uint8)
no_bk = cv2.erode(no_bk, kernel_2, iterations=2)
no_bk = cv2.dilate(no_bk, kernel_4, iterations=2)

cv2.imwrite('no_noise.jpg', no_bk)

# edged = cv2.Canny(no_bk, 230, 250)

# cv2.imwrite('edged.jpg', edged)

ROI_number = 0
contours, hierarchy = cv2.findContours(no_bk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print('Number of contours: ' + str(len(contours)))
for c in contours:
    # Obtain bounding rectangle to get measurements
    x,y,w,h = cv2.boundingRect(c)

    # Find centroid
    M = cv2.moments(c)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cX, cY = 0, 0

    # Crop and save ROI
    ROI = original[y:y+h, x:x+w]
    #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1

    # Draw the contour and center of the shape on the image
    cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12), 4)
    cv2.circle(img, (cX, cY), 10, (320, 159, 22), -1) 

cv2.imwrite('image.png', img)


# # Processing data
# arr = np.zeros(shape = (len(contours)))
# for i in range(len(contours)):
#     area = cv2.contourArea(contours[i])
#     arr[i] = area
#     mean_arr = np.mean(arr)
# count = 0
# result_img = rgb_img.copy()
# for i in range(len(contours)):
#     # Getting the center coordinate of the contour
#     m = cv2.moments(contours[i])
#     center_x = int(m["m10"]/m["m00"])
#     center_y = int(m["m01"]/m["m00"])
#     # Counting and displaying counted object in circle
#     if arr[i] < 1.65*mean_arr:
#         cv2.circle(result_img, (center_x, center_y), radius = 20, color = (0,255,0),thickness = 2)
#         count += 1
#     else:
#         for j in range(2):
#             cv2.circle(result_img,(center_x, center_y),radius = 20+10*j,color = (0,255,0),thickness = 2)
#             count += 2
# cv2.imwrite('circled.jpg', result_img)