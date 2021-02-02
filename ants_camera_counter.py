import cv2
import matplotlib.pyplot
import time
import numpy as np

from datetime import datetime

#USER DEFINED PARAMETERS

IPaddressOnRemotePi = '130.64.143.183'

USBpathOnThisPi = '/media/pi/0012-D687/'

USBpathOnRemotepi = '/media/pi/CRUZR/'

#End of USER DEFINED PARAMETERS

# initialize the camera

#cam = cv2.VideoCapture(0)

fps = 5

cam = cv2.VideoCapture('/dev/video0')

#print("here\n")

#https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html

#look at this link to change properties of the logitech 920 camera

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 240)

cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#USER DEFINED PARAMETERS

#depending on where you set the camera relative to the antbox, you may need to adjust the focus of the camera, i.e 0.1 to some other value betwee 0-1

cam.set(cv2.CAP_PROP_FOCUS, 0.1)

#End of USER DEFINED PARAMETERS

from paramiko import SSHClient

from scp import SCPClient

ssh = SSHClient()

ssh.load_system_host_keys()

ssh.connect(IPaddressOnRemotePi, username='pi', password='raspberry')

##SCPClient takes a pramiko transport as an argument

scp = SCPClient(ssh.get_transport())

while True:

#print("here\n")

    start_time = time.time()

    ret, image = cam.read()

    if ret:

        stringTime = datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')

        print('last save imgage: ' + stringTime)

        img_path = USBpathOnThisPi + stringTime+ '.jpg'

        cv2.imwrite(img_path,image)

        scp.put(img_path, recursive=False, remote_path=USBpathOnRemotepi + stringTime+ '.jpg')

        scp.get('/home/pi/Pictures',recursive=True, local_path='/home/pi/Pictures/*')

        ##Opening the image, converting from BGR to RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Unreadable image. Check the image path\n")
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #Converting to HSV for image analysis. This snippet of code was taken 
            #from the pumpkin counting Medium article.
            # Declaring some variables
            MIN,LOW,HUE = 0, 0, 0
            MAX,HIGH,SAT = 1, 1, 1
            VAL = 2
            # Tuple for storing HSV range
            HSV_Range = ((0, 0, 0), (179, 255, 255))
            # List for storing HSV values
            HSV_Values = [[20, 170, 170], [179, 255, 255]]
            # Storing HSV values in a scalar array
            lower_HSV = np.array([HSV_Values[LOW][HUE],HSV_Values[LOW][SAT],HSV_Values[LOW][VAL]])
            upper_HSV = np.array([HSV_Values[HIGH][HUE],HSV_Values[HIGH][SAT],HSV_Values[HIGH][VAL]])

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
            processed_img = cv2.inRange(hsv_img, lower_HSV, upper_HSV)

            #Removing the noise/grains
            kernel_4 = np.ones((4,4),np.uint8)
            kernel_2 = np.ones((2,2),np.uint8)
            processed_img = cv2.erode(processed_img, kernel_2, iterations=2)
            processed_img = cv2.dilate(processed_img, kernel_4, iterations=2)

            #Counting the ants by using contours
            contours, hierarchy = cv2.findContours(processed_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            print("Ants count: {}".format(len(contours))) # This could be inaccurate if some of the ants are too close togehter 
            # Processing data
            arr = np.zeros(shape = (len(contours)))
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                arr[i] = area  
            mean_arr = np.mean(arr)
            count = 0
            result_img = rgb_img.copy()
            for i in range(len(contours)):
                # Getting the center coordinate of the contour
                m = cv2.moments(contours[i])
                center_x = int(m["m10"]/m["m00"])
                center_y = int(m["m01"]/m["m00"])
                # Counting and displaying counted object in circle
                if arr[i] < 1.65*mean_arr:
                    cv2.circle(result_img,(center_x, center_y),radius = 20,color = (0,255,0),thickness = 2)
                    count+=1
                else:
                    for j in range(2):
                        cv2.circle(result_img,(center_x, center_y),radius = 20+10*j,color = (0,255,0),thickness = 2)
                        count += 2

            
            # Displaying number of ants
            
            text = "Object count: " + str(count) 
            # Resizing the canvas
            fig = plt.figure(figsize = (20,20))
            ax = fig.add_subplot(111)
            # Displaying
            ax.set_title('Result')
            ax.imshow(result_img)    
        
            elapsed_time = time.time() - start_time

            print((1/fps) - elapsed_time)

    try:

        time.sleep( (1/fps) - elapsed_time )

    except:

        print('negative sleep time')

        print('Due to slow scp transfer of data from this CPU to remote CPU, your fps=' + str(fps) +' is not achievable ')

    cam.release()

    scp.close()
