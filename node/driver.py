#! /usr/bin/env python3

import sys
import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import rosgraph_msgs
import time
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from tensorflow.compat.v1 import ConfigProto
import cv2
import numpy as np
from geometry_msgs.msg import Twist
# For Mack PC
# !export "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda/"
# config = ConfigProto()
# config.gpu_options.allow_growth = True

class comp_driver:
    OUTER_LOAD_PATH = "/home/fizzer/cnn_trainer/model_save/"
    INNER_LOAD_PATH = "/home/fizzer/cnn_trainer/inner/model_save/"

    def __init__(self):
        self.state = "outer"
        self.ped_count = 0
        self.startup_time = time.time()
        self.biggest_plate_size = 0
        self.analyzed = True
        
        self.controller = driver_controller(self.OUTER_LOAD_PATH, lin_speed=0.4)
        self.inner_controller = driver_controller(self.INNER_LOAD_PATH, lin_speed=0.5)

        self.mover = rospy.Publisher("/R1/cmd_vel",
                                        Twist,
                                        queue_size = 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",
                                            Image,
                                            self.callback)
        self.licenses = rospy.Publisher("/license_plate",
                                            String,
                                            queue_size = 4)
        #self.timer = rospy.Subscriber("/clock",
        #                                rosgraph_msgs.Clock) 
        time.sleep(1)

        self.pub = rospy.Subscriber('comms', String, self.ped_callback)
        # self.startup_time = self.timer.
        self.licenses.publish("TeamEthan,notsafe,0,AA00") #This should start the timer, ask Miti what license plate number to use
        self.timer_running = True

    def move_bot(self, move_command):
        try:
            self.mover.publish(move_command)
        except CvBridgeError as e:
            print(e)

    def seek_license(self):
        TOP_CUT = 350
        BOTTOM_CUT = 550
        LOWER_THRESHOLD = 90
        UPPER_THRESHOLD = 210
        MIN_CONTOUR_AREA = 11000
    
        cut_image = self.raw_cv_image[TOP_CUT:BOTTOM_CUT,:]
        hsv_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2HSV)
        blur_image = cv2.GaussianBlur(hsv_image, (5,5), 0)
        threshold_image = cv2.inRange(blur_image, np.array([0,0,LOWER_THRESHOLD]), np.array([0,0,UPPER_THRESHOLD]))
    
        # new_threshold = threshold_image.copy()
    
        # cv2.imshow("Threshold feed", threshold_image)
        # cv2.waitKey(3)
    
        contours, _ = cv2.findContours(threshold_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    
        approx = None
        marked_raw = self.raw_cv_image
    
        if len(cntsSorted) > 0 and cv2.contourArea(cntsSorted[-1]) > MIN_CONTOUR_AREA:
            
            max_contour = cntsSorted[-1]
            
            epsilon = 0.01*cv2.arcLength(max_contour,True)
            approx = cv2.approxPolyDP(max_contour,epsilon, True)
    
            if len(approx) is 4:
    
                # new_threshold = cv2.cvtColor(new_threshold, cv2.COLOR_GRAY2BGR)
                for i in range(len(approx[:,0,1])):
    
                    approx[i,0,1] = approx[i,0,1] + TOP_CUT
    
                marked_raw = cv2.drawContours(self.raw_cv_image, approx, -1, (0,0,255), 5)
                # cv2.imshow("Seen Contour", marked_raw)
                # cv2.waitKey(3)
    
        if len(cntsSorted) > 0:
            print(cv2.contourArea(cntsSorted[-1]))
            print("----")
        # cv2.imshow("Marked raw feed", marked_raw)
        # cv2.waitKey(3)
    
        return approx
    
    
    def check_plate(self, approx):
        print (self.analyzed)
        print(self.biggest_plate_size)
        if approx is None and self.analyzed is False:
            self.analyze_plate(self.biggest_plate)
            self.analyzed = True
        elif approx is None and self.analyzed is True:
            pass
        else:
            if self.analyzed is True:
                self.analyzed = False
                self.biggest_plate_size = 0
    
            if cv2.contourArea(approx) > self.biggest_plate_size and len(approx) == 4:
                self.biggest_plate = approx
                self.best_plate_image = self.raw_cv_image
                self.biggest_plate_size = cv2.contourArea(approx)
    
    # This function utalizes https://arccoder.medium.com/straighten-an-image-of-a-page-using-opencv-313182404b06
    def analyze_plate(self,approx):
        sortedApprox = sorted(approx, key = lambda x: x[0,0] + 5*x[0,1])
        
        height = 500
        width = 500
    
        finalPoints = [[0,0],[width,0],[0,height],[width,height]]
        M = cv2.getPerspectiveTransform(np.float32(sortedApprox), np.float32(finalPoints))
        dst = cv2.warpPerspective(self.best_plate_image, M, (int(width),int(height)+150))
    
        cv2.imshow("dst feed", dst)
        cv2.waitKey(3)
    
        plate_height = 150
        plate_image = dst[dst.shape[0] - plate_height:,:]
        cv2.imshow("plate feed", plate_image)
        cv2.waitKey(3)

    def state_machine(self):
        print(self.state)

        if self.state == "startup":
            move_command = Twist()

            if(time.time() < self.startup_time + 4):
                move_command.linear.x = 0.2
                move_command.angular.z = 0.0
            elif(time.time() < self.startup_time + 7.4):
                move_command.linear.x = 0.0
                move_command.angular.z = 0.6
            else:
                move_command.linear.x = 0
                move_command.angular.z = 0
                self.state = 'outer'

            self.move_bot(move_command)

        elif self.state == "outer":
            #Implement imitation learning mover
            #Implement license plate detector and reader
            #Implement licence counter to know when to switch to seeking inner
            #Implement pedestrian seeker
            move_command = self.controller.drive(self.raw_cv_image)
            self.move_bot(move_command)
            license_corners = self.seek_license()   
            self.check_plate(license_corners)

            # if time.time() > self.startup_time + 100:
            #    self.state = "terminate"

        elif self.state == "ped_stop":
            move = Twist()
            move.linear.x = 0.0
            move.angular.z = 0.0
            self.move_bot(move)
        elif self.state == "ped_drive":
            move = Twist()
            move.linear.x = 0.5
            move.angular.z = 0.0
            self.move_bot(move)
            time.sleep(1.0)
            if self.ped_count:
                self.state = "inner"
            else:
                self.ped_count += 1
                self.state = "outer"
        
        elif self.state == "inner":
            move_command = self.inner_controller.drive(self.raw_cv_image)
            self.move_bot(move_command)
            license_corners = self.seek_license()
            self.check_plate(license_corners)

        elif self.state == "terminate":

            if self.timer_running:
                move = Twist()
                move.linear.x = 0
                move. angular.z = 0
                self.move_bot(move)
                self.licenses.publish("TeamEthan,notsafe,-1,AA00")
                self.timer_running = False

    def callback(self,data):
        try:
            self.raw_cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e) #Shouldn't ever get here cause callback gets called with an image
        self.state_machine()
    
    def ped_callback(self,data):
        msg = data.data
        if msg == "stop":
            print("STOP\n")
            self.state = "ped_stop"
        elif msg == "ped_drive":
            self.state = "ped_drive"

class driver_controller:
    SAVE_PATH = "/home/fizzer/cnn_trainer/model_save/"
    IMG_DOWNSCALE_RATIO = 0.25

    def __init__(self, save_path = SAVE_PATH, lin_speed = 0.2, ang_speed = 1.0) -> None:
        self.one_hot_ref = {
            'L' : np.array([1.,0.,0.]),
            'F' : np.array([0.,1.,0.]),
            'R' : np.array([0.,0.,1.]),
        }
        self.conv_model = models.load_model(save_path)
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed

    def drive(self, img):
        img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0,0),
                            fx=self.IMG_DOWNSCALE_RATIO, fy=self.IMG_DOWNSCALE_RATIO)
        img = img.reshape(1, len(img), len(img[0]), -1)
        prediction = self.conv_model.predict(img)[0]
        print(prediction)
        move = Twist()
        if round(prediction[0]) == 1.:
            move.linear.x = 0.0
            move.angular.z = self.ang_speed
        elif round(prediction[1]) == 1.:
            move.linear.x = self.lin_speed
            move.angular.z = 0.0
        elif round(prediction[2]) == 1.:
            move.linear.x = 0.0
            move.angular.z = -1 * self.ang_speed
        else:
            # TODO: REMOVE (??)
            print("Error - invalid command")
            move.linear.x = 0.1
            move.angular.z = 0.0
        print(f"x: {move.linear.x}\nz: {move.angular.z}")
        return move

rospy.init_node('comp_driver', anonymous = True)
comp = comp_driver()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()