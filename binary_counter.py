import cv2
import matplotlib.pyplot
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Read BGR image
img = cv2.imread('2_ants.jpg')
background = cv2.imread('bkgrnd.jpg')

# creating object 
#fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();    
#fgbg2 = cv2.createBackgroundSubtractorMOG2(); 
#fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG(); 


if img is None or background is None:
            raise Exception("Unreadable image. Check the image path\n")




height, width, channels = img.shape

# # Compute SSIM between two images
# (score, diff) = ssim(img, background, full=True, multichannel=True)
# print("Image similarity", score)

# # The diff image contains the actual image differences between the two images
# # and is represented as a floating point data type in the range [0,1] 
# # so we must convert the array to 8-bit unsigned integers in the range
# # [0,255] before we can use it with OpenCV
# diff = (diff * 255).astype("uint8")

# # Threshold the difference image, followed by finding contours to
# # obtain the regions of the two input images that differ
# thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# mask = np.zeros(before.shape, dtype='uint8')
# filled_after = after.copy()

# Creating grayscale image
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bkgrnd_grayscale = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)

# # Convert grayscale to binary
ret, img_binary = cv2.threshold(img_grayscale,127,255,cv2.THRESH_BINARY)
ret, bkgrnd_binary = cv2.threshold(bkgrnd_grayscale, 127, 255, cv2.THRESH_BINARY)
# #Writing the image
cv2.imwrite('image_binary.jpg',img_binary)
cv2.imwrite('bk_binary.jpg', bkgrnd_binary)

backSub = cv2.createBackgroundSubtractorMOG2()
mask = backSub.apply(img)
sub = cv2.subtract(img, background)
cv2.imwrite('without_background.jpg', sub)



#Counting based on contours

#Removing the noise/grains
kernel_4 = np.ones((4,4),np.uint8)
kernel_2 = np.ones((2,2),np.uint8)
sub = cv2.erode(sub, kernel_2, iterations=2)
sub = cv2.dilate(sub, kernel_4, iterations=2)
 
cv2.imwrite('noise_erased.jpg', sub)

#Counting the ants by using contours
contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Ants count: {}".format(len(contours))) # This could be inaccurate if some of the ants are too close togehter 
# Processing data
arr = np.zeros(shape = (len(contours)))
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    arr[i] = area  
mean_arr = np.mean(arr)
count = 0



cv2.imwrite('circled.jpg', sub)