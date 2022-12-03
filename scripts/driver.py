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

    def __init__(self):

        self.state = "startup"
        
        self.controller = driver_controller()

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
        self.timer = rospy.Subscriber("/clock",
                                        rosgraph_msgs.Clock) 
        time.sleep(1)

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
        MIN_CONTOUR_AREA = 9500

        cut_image = self.raw_cv_image[TOP_CUT:BOTTOM_CUT,:]
        hsv_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2HSV)
        blur_image = cv2.GaussianBlur(hsv_image, (5,5), 0)
        threshold_image = cv2.inRange(blur_image, np.array([0,0,LOWER_THRESHOLD]), np.array([0,0,UPPER_THRESHOLD]))

        new_threshold = threshold_image.copy()

        cv2.imshow("Threshold feed", threshold_image)
        cv2.waitKey(3)

        contours, hierarchy = cv2.findContours(new_threshold.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        approx = None
        marked_raw = self.raw_cv_image

        if len(cntsSorted) > 0 and cv2.contourArea(cntsSorted[-1]) > MIN_CONTOUR_AREA:
            
            max_contour = cntsSorted[-1]
            
            epsilon = 0.01*cv2.arcLength(max_contour,True)
            approx = cv2.approxPolyDP(max_contour,epsilon, True)

            if len(approx) == 4:

                new_threshold = cv2.cvtColor(new_threshold, cv2.COLOR_GRAY2BGR)
                
                    
                for i in range(len(approx[:,0,1])):

                    approx[i,0,1] = approx[i,0,1] + TOP_CUT

                marked_raw = cv2.drawContours(self.raw_cv_image, approx, -1, (0,0,255), 5)

                print(cv2.contourArea(cntsSorted[-1]))
                print("----")

                cv2.imshow("Seen Contour", marked_raw)
                cv2.waitKey(3)

        
        cv2.imshow("Marked raw feed", marked_raw)
        cv2.waitKey(3)

        return approx


    def state_machine(self):

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

            if time.time() > self.startup_time + 100:
                self.state = "terminate"

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
        
        # cv2.imshow("raw feed", self.raw_cv_image)
        # cv2.waitKey(3)

class driver_controller:
    CP_PATH = "/home/fizzer/cnn_trainer/model_cps/"
    SAVE_PATH = "/home/fizzer/cnn_trainer/model_save/"
    MODEL_X = 180
    MODEL_Y = 320
    LEARNING_RATE = 1e-4
    IMG_DOWNSCALE_RATIO = 0.25

    def __init__(self, save_path = SAVE_PATH) -> None:
        self.one_hot_ref = {
            'L' : np.array([1.,0.,0.]),
            'F' : np.array([0.,1.,0.]),
            'R' : np.array([0.,0.,1.]),
        }
        self.conv_model = models.load_model(save_path)
        # self.conv_model = self.__create_model()
        # self.conv_model.load_weights(self.CP_PATH)

    def drive(self, img):
        img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0,0),
                            fx=self.IMG_DOWNSCALE_RATIO, fy=self.IMG_DOWNSCALE_RATIO)
        img = img.reshape(1, len(img), len(img[0]), -1)
        prediction = self.conv_model.predict(img)[0]
        move = Twist()
        if prediction[0] == 1.:
            move.linear.x = 0.0
            move.angular.z = 0.7
        elif prediction[1] == 1.:
            move.linear.x = 0.3
            move.angular.z = 0.0
        elif prediction[2] == 1.:
            move.linear.x = 0.0
            move.angular.z = -0.7
        else:
            # TODO: REMOVE (??)
            move.linear.x = 0.0
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