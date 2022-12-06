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
    IMG_RESIZE = 0.1
    VERTICAL_SLICE = ()
    MIN_CNT_AREA = 5.0
    PED_CENTER_REGION = (400.,480.)
    MASK_FRAMES = 40
    MIN_PED_AREA = 200.0
    MIN_MOVE_DIST = 10
    
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
        cv2.imshow("mask" , fg_mask)
        cv2.waitKey(1)
        return fg_mask
    
    def img_callback(self, data):
        # print(self.state)
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

        if self.state == States.FIND_LINE or self.state == States.PREP_STOP: 
            downsize = cv2.resize(img, (0,0), fx = self.IMG_RESIZE, fy = self.IMG_RESIZE)
            hsv = cv2.cvtColor(downsize, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            contours, _ = cv2.findContours(mask, 
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            if len(contours) > 0 and cv2.contourArea(cnts_sorted[-1]) > self.MIN_CNT_AREA:
                # cv2.imshow("Red Mask", mask)
                # cv2.waitKey(1)
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
                print(cv2.contourArea(cnts_sorted[-1]))
                moment = cv2.moments(cnts_sorted[-1])
                cX = int((moment["m10"]+0.00001) / (moment["m00"]+0.00001))
                print(cX, " ", self.prev_cX)
                if self.PED_CENTER_REGION[0] <= cX <= self.PED_CENTER_REGION[1]:
                    print("ped in center")
                    self.state = States.WAIT_PED
                    time.sleep(0.2)
                elif self.state == States.WAIT_PED and abs(cX - self.prev_cX) < self.MIN_MOVE_DIST:
                    self.state = States.DRIVE
                self.prev_cX = cX
        
        if self.state == States.DRIVE:
            self.pub.publish("ped_drive")
            time.sleep(15)
            if not self.first_crosswalk:
                self.state = States.FIND_LINE
                self.first_crosswalk = True
                self.bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            else:
                self.state = States.PREP_INNER

                # cY = int(moment["m01"] / moment["m00"])
                # print(img.shape)
                # cv2.circle(img, (cX,cY), 3, (255,0,0), -1)
                # cv2.imshow("mask", fg_mask)
                # cv2.imshow("dot", img)
                # cv2.waitKey(1)


class States(Enum):
    FIND_LINE = auto()
    PREP_STOP = auto()
    BUILD_MASK = auto()
    TRACK_PED = auto()
    WAIT_PED = auto()
    DRIVE = auto()
    PREP_INNER = auto()
        
rospy.init_node('ped_handler', anonymous = True)
ped = ped_handler()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()
