#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from enum import Enum, auto
import time

class ped_handler:
    HSV_THRESH = {
        "uh": 5,
        "us": 255,
        "uv": 255,
        "lh": 0,
        "ls": 250,
        "lv": 250
    }
    IMG_RESIZE = 0.1 # The scaling factor for downsampling the image when detecting the red line
    MIN_CNT_AREA = 5.0 # The minimum size of the contour for detecting/stopping at the red line
    MIN_CAR_AREA = 150.0 # The minimum size of the counter for detecting/stopping before the inner loop
    MIN_PED_AREA = 200.0 # The minimum size of the contour in the background subtracted image that we will consider to be the pedestrian
    MIN_TRUCK_AREA = 2000.0 # The minimum size of the contour in the background subtracted image that we will consider to be the pedestrian
    PED_CENTER_REGION = (400.,480.) # The bounds on the width we will check within for the pedestrian
    TRUCK_CENTER_REGION = (400.,480.) # The bounds on the width we will check within for the truck
    MASK_FRAMES = 30 # The number of frames we will train the background mask on before attempting to enter the intersection
    MIN_MOVE_DIST = 10 # The number of pixels we will allow the pedestrian to move before we consider it to not be stationary
    
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",
                                            Image,
                                            self.img_callback)
        self.pub = rospy.Publisher('comms', String, queue_size=10)
        self.lower_hsv = np.array([self.HSV_THRESH["lh"], self.HSV_THRESH["ls"],self.HSV_THRESH["lv"]])
        self.upper_hsv = np.array([self.HSV_THRESH["uh"], self.HSV_THRESH["us"],self.HSV_THRESH["uv"]])
        self.state = States.FIND_LINE
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.mask_count = 0
        self.prev_cX = 0
        self.consecutive_dropped_line = 0
        self.first_crosswalk = False
    
    def update_mask(self, img):
        # img = cv2.resize(img, (0,0), fx = self.IMG_RESIZE, fy = self.IMG_RESIZE)
        img = img[200:520, 200:1080]
        fg_mask = self.bg_sub.apply(img, learningRate = -1)
        # cv2.imshow("mask" , fg_mask)
        # cv2.waitKey(1)
        return fg_mask

    def filter_img(self, img):
        downsize = cv2.resize(img, (0,0), fx = self.IMG_RESIZE, fy = self.IMG_RESIZE)
        hsv = cv2.cvtColor(downsize, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
    
    def img_callback(self, data):
        # print(self.state)
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

        if self.state == States.FIND_LINE or self.state == States.PREP_STOP: 
            mask = self.filter_img(img)
            
            contours, _ = cv2.findContours(mask, 
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            if len(contours) > 0 and cv2.contourArea(cnts_sorted[-1]) > self.MIN_CNT_AREA:
                self.state = States.PREP_STOP
                self.consecutive_dropped_line = 0
            elif self.state == States.PREP_STOP and self.consecutive_dropped_line < 2:
                self.consecutive_dropped_line += 1
            elif self.state == States.PREP_STOP:
                self.pub.publish("stop")
                self.state = States.BUILD_MASK

        if self.state == States.BUILD_MASK:
            fg_mask = self.update_mask(img)
            self.mask_count += 1
            if self.mask_count > self.MASK_FRAMES:
                self.state = States.TRACK_PED
                self.mask_count = 0

        if self.state == States.TRACK_PED or self.state == States.WAIT_PED:
            fg_mask = self.update_mask(img) 
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            if len (contours) > 0 and cv2.contourArea(cnts_sorted[-1]) > self.MIN_PED_AREA:
                # print(cv2.contourArea(cnts_sorted[-1]))
                moment = cv2.moments(cnts_sorted[-1])
                cX = int((moment["m10"]+0.00001) / (moment["m00"]+0.00001))
                print(cX, " ", self.prev_cX)
                if self.PED_CENTER_REGION[0] <= cX <= self.PED_CENTER_REGION[1]:
                    print("Pedestrian in center")
                    self.state = States.WAIT_PED
                    time.sleep(0.5)
                    self.state = States.DRIVE
        
        if self.state == States.DRIVE:
            self.pub.publish("ped_drive")
            time.sleep(1)
            if not self.first_crosswalk:
                self.state = States.FIND_LINE
                self.first_crosswalk = True
                self.bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            else:
                self.lower_hsv = np.array([119,206,114])
                self.upper_hsv = np.array([120,217,126])
                self.bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
                self.state = States.PREP_INNER
                time.sleep(3)
        
        if self.state == States.PREP_INNER:
            # TODO: Remove vvv
            mask = self.filter_img(img)
            mask = mask[:,len(mask)*2//3:]
            # cv2.imshow("test", img[:,len(img)//2:])
            # cv2.waitKey(1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            if len(contours) > 0 and cv2.contourArea(cnts_sorted[-1]) > self.MIN_CAR_AREA:
                # print(f"Car contour:{cv2.contourArea(cnts_sorted[-1])}")
                self.pub.publish("stop")
                self.state = States.TRUCK_MASK
                time.sleep(0.3)
        
        if self.state == States.TRUCK_MASK:
            fg_mask = self.update_mask(img)
            self.mask_count += 1
            if self.mask_count > self.MASK_FRAMES:
                self.state = States.WAIT_TRUCK
                self.mask_count = 0
        
        if self.state == States.WAIT_TRUCK:
            fg_mask = self.update_mask(img)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            # print(f"Truck mask countour: {cv2.contourArea(cnts_sorted[-1])}")
            if len (contours) > 0 and cv2.contourArea(cnts_sorted[-1]) > self.MIN_TRUCK_AREA:
                # print(cv2.contourArea(cnts_sorted[-1]))
                moment = cv2.moments(cnts_sorted[-1])
                cX = int((moment["m10"]+0.00001) / (moment["m00"]+0.00001))
                print(cX)
                if self.TRUCK_CENTER_REGION[0] <= cX <= self.TRUCK_CENTER_REGION[1]:
                    print("Truck in center")
                    time.sleep(0.5)
                    self.state = States.DRIVE_INNER
                    self.pub.publish("drive_inner")

class States(Enum):
    FIND_LINE = auto()
    PREP_STOP = auto()
    BUILD_MASK = auto()
    TRACK_PED = auto()
    WAIT_PED = auto()
    DRIVE = auto()
    PREP_INNER = auto()
    WAIT_TRUCK = auto()
    TRUCK_MASK = auto()
    DRIVE_INNER = auto()
        
rospy.init_node('ped_handler', anonymous = True)
ped = ped_handler()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()
