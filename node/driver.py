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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
import cv2
import numpy as np
from geometry_msgs.msg import Twist

class comp_driver:
    OUTER_LOAD_PATH = "/home/fizzer/cnn_trainer/model_save/"
    INNER_LOAD_PATH = "/home/fizzer/cnn_trainer/inner/model_save/"

    def __init__(self):
        self.state = "outer"
        self.ped_count = 0
        self.startup_time = time.time()

        self.controller = driver_controller(self.OUTER_LOAD_PATH, lin_speed=0.40, ang_speed=0.90)
        self.inner_controller = driver_controller(self.INNER_LOAD_PATH, lin_speed=0.40, ang_speed=0.90)

        time.sleep(3)
        
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

        self.licenses.publish("TeamEthan,notsafe,0,AA00") 
        self.timer_running = True

    def move_bot(self, move_command):
        try:
            self.mover.publish(move_command)
        except CvBridgeError as e:
            print(e)

    def state_machine(self):
        # print(f"Drive state: {self.state}")
        if self.state == "startup":
            # This was unused in competition, it was an early debug state we used to make a specific maneuver at initialization
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
            move_command = self.controller.drive(self.raw_cv_image)
            self.move_bot(move_command)

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
                del(self.controller)
                self.state = "inner"
            else:
                self.ped_count += 1
                self.state = "outer"
        
        elif self.state == "inner":
            move_command = self.inner_controller.drive(self.raw_cv_image)
            self.move_bot(move_command)

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
        elif msg == "drive_inner":
            self.state = "inner"

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
            print("Error - invalid command")
            move.linear.x = 0.1
            move.angular.z = 0.0
        print(f"x: {move.linear.x}\nz: {move.angular.z}")
        return move

# For Mack PC
# Comment out if not using GPU for networks
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
rospy.init_node('comp_driver', anonymous = True)
comp = comp_driver()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()