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
        # TODO: remove:
        self.biggest_plate_size = 0
        self.analyzed = True

        #TODO: remove
        # self.plate_path = "/home/fizzer/plate_images"
        # self.counter = self.set_im_num()

        self.controller = driver_controller(self.OUTER_LOAD_PATH, lin_speed=0.20, ang_speed=0.5)
        self.inner_controller = driver_controller(self.INNER_LOAD_PATH, lin_speed=0.30, ang_speed=0.5)

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
    
        contours, _ = cv2.findContours(threshold_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    
        approx = None
        # marked_raw = self.raw_cv_image
    
        if len(cntsSorted) > 0 and cv2.contourArea(cntsSorted[-1]) > MIN_CONTOUR_AREA:
            
            max_contour = cntsSorted[-1]
            
            epsilon = 0.01*cv2.arcLength(max_contour,True)
            approx = cv2.approxPolyDP(max_contour,epsilon, True)
    
            if len(approx) == 4:
    
                for i in range(len(approx[:,0,1])):
    
                    approx[i,0,1] = approx[i,0,1] + TOP_CUT
    
                # marked_raw = cv2.drawContours(self.raw_cv_image, approx, -1, (0,0,255), 5)
                # cv2.imshow("Seen Contour", marked_raw)
                # cv2.waitKey(3)
    
        # if len(cntsSorted) > 0:
        #     print(f"Plate contour: {cv2.contourArea(cntsSorted[-1])}")
        #     print("----")
        # cv2.imshow("Marked raw feed", marked_raw)
        # cv2.waitKey(3)
    
        return approx
    
    
    def check_plate(self, approx):
        # print (self.analyzed)
        # print(self.biggest_plate_size)
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
    
    # This function utilizes https://arccoder.medium.com/straighten-an-image-of-a-page-using-opencv-313182404b06
    def analyze_plate(self,approx):
        sortedApprox = sorted(approx, key = lambda x: x[0,0] + 5*x[0,1])
        
        height = 500
        width = 500
    
        finalPoints = [[0,0],[width,0],[0,height],[width,height]]
        M = cv2.getPerspectiveTransform(np.float32(sortedApprox), np.float32(finalPoints))
        dst = cv2.warpPerspective(self.best_plate_image, M, (int(width),int(height)+150))

        # This was just being used to display and save the licence plates
        # cv2.imshow("dst feed", dst)
        # cv2.waitKey(3)
    
        plate_height = 150
        plate_image = dst[dst.shape[0] - plate_height:,:]

        # This was just being used to display and save the licence plates
        # cv2.imshow("plate feed", plate_image)
        # cv2.waitKey(3)
        # # TODO: remove
        # cv2.imwrite(f"{self.plate_path}/PLT_{self.counter:06d}.jpg", plate_image)
        # self.counter +=1

        self.make_CNN_chars(dst, plate_image)
    def make_CNN_chars(self,full_image, plate_image):
        plate_lower = np.array([115,80,90])
        plate_upper = np.array([122,255,205])
        full_lower = np.array([0,0,0])
        full_upper = np.array([0,0,90])
    
        hsv_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
        hsv_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        mask_full = cv2.inRange(hsv_full, full_lower, full_upper)
        mask_plate = cv2.inRange(hsv_plate, plate_lower, plate_upper)
    
        cv2.imshow("cut plate feed", mask_plate)
        cv2.waitKey(3)
        cv2.imshow("full plate feed", mask_full)
        cv2.waitKey(3)
 
       #TODO: pass both the license plate identifier number image and a
       #list of the different characters on the plate images to
       #read_licence
    
    def read_licence(self, plate_identifier_image, plate_char_images):
        #TODO: initialize self.submission_timer to 0 in whatever object it belongs to
    
        encoder_options = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        encoder_len = len(encoder_options)
    
        def zero_hot(len):
            return np.zeros(len, dtype=int)
    
        if time.time() > self.submission_timer + 2:
    
            identity = encoder_options[np.argmax(self.plate_identifier_guesses)]
            char0 = encoder_options[np.argmax(self.plate_char_guesses[0])]
            char1 = encoder_options[np.argmax(self.plate_char_guesses[1])]
            char2 = encoder_options[np.argmax(self.plate_char_guesses[2])]
            char3 = encoder_options[np.argmax(self.plate_char_guesses[3])]
    
            
    
            #This exists in driver.py so this call needs to link to it somehow
            self.licenses.publish(str('TeamEthan,notsafe,' + identity + ',' + char0
                                                                            + char1
                                                                            + char2
                                                                            +char3))
    
            self.plate_identifier_guesses = zero_hot(encoder_len)
            self.plate_char_guesses = [zero_hot(encoder_len),
                                        zero_hot(encoder_len),
                                        zero_hot(encoder_len),
                                        zero_hot(encoder_len)]
        
        #CNN here is meant to pass the thing in brackets to the CNN and get pack a 36 long 1 hot array of the CNN's guess
        plate_identifier_guess = CNN(plate_identifier_image)
        plate_char_guesses = CNN(plate_char_images)
    
        self.plate_identifier_guesses += plate_identifier_guess
    
        for char, i in enumerate(plate_char_guesses):
            self.plate_char_guesses[i] += char
    
        self.submission_timer = time.time()

    
    # #TODO: REMOVE
    # def set_im_num(self):
    #     file_list = [s[-10:-4] for s in os.listdir(self.plate_path)]
    #     list.sort(file_list)
    #     if (len(file_list) > 0):
    #         print(f"Previously saved images detected - starting from image {int(file_list[-1])}")
    #         return int(file_list[-1])
    #     else:
    #         return 0

    def state_machine(self):
        # print(f"Drive state: {self.state}")
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
            move_command = self.controller.drive(self.raw_cv_image)
            self.move_bot(move_command)
            # license_corners = self.seek_license()   
            # self.check_plate(license_corners)

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
            # license_corners = self.seek_license()
            # self.check_plate(license_corners)

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

#TODO: remove
class plate_reader:
    SAVE_PATH = "/home/fizzer/cnn_trainer/letter_model/save/"

    def __init__(self, save_path = SAVE_PATH):
        self.conv_model = models.load_model(save_path)
    
    def predict(self, img):
        return self.conv_model.predict(img)[0]

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