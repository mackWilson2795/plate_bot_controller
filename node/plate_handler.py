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
from enum import Enum, auto

class plate_handler:
    TOP_CUT = 350
    BOTTOM_CUT = 550
    LOWER_THRESHOLD = 90
    UPPER_THRESHOLD = 210
    MIN_CONTOUR_AREA = 11000
    KERNEL = np.ones((3,3), np.uint8)
    ONE_HOT_REF = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    CHAR_DIMS = (75,100)
    DST_PATH = "/home/fizzer/cnn_trainer/dst_model/save/"

    def __init__(self):
        time.sleep(3)
        self.plate_reader = plate_reader()
        self.dst_reader = plate_reader()
        self.plate_lower = np.array([115,80,90])
        self.plate_upper = np.array([122,255,205])
        self.full_lower = np.array([0,0,0])
        self.full_upper = np.array([0,0,90])
        self.biggest_plate_size = 0
        self.analyzed = True
        self.state = States.FIND_PLATES
        self.submission_timer = 0

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",
                                            Image,
                                            self.callback)
        self.licenses = rospy.Publisher("/license_plate",
                                            String,
                                            queue_size = 4)
        self.pub = rospy.Subscriber('comms', String, self.ped_callback)

    def callback(self, data):
        try:
            self.raw_cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.state_machine()

    def ped_callback(self,data):
        msg = data.data
        if msg == "stop":
            self.state = States.WAIT
        else:
            self.state = States.FIND_PLATES
    
    def state_machine(self):
        print(self.state)
        if self.state == States.FIND_PLATES:
            license_corners = self.seek_license()
            self.check_plate(license_corners)
        elif self.state == States.WAIT:
            pass

    def seek_license(self):
        cut_image = self.raw_cv_image[self.TOP_CUT:self.BOTTOM_CUT,:]
        hsv_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2HSV)
        blur_image = cv2.GaussianBlur(hsv_image, (5,5), 0)
        threshold_image = cv2.inRange(blur_image, np.array([0,0,self.LOWER_THRESHOLD]), np.array([0,0,self.UPPER_THRESHOLD]))
        contours, _ = cv2.findContours(threshold_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        approx = None
        if len(cntsSorted) > 0 and cv2.contourArea(cntsSorted[-1]) > self.MIN_CONTOUR_AREA:
            max_contour = cntsSorted[-1]
            epsilon = 0.01*cv2.arcLength(max_contour,True)
            approx = cv2.approxPolyDP(max_contour,epsilon, True)
            if len(approx) == 4:
                for i in range(len(approx[:,0,1])):
                    approx[i,0,1] = approx[i,0,1] + self.TOP_CUT
        return approx


    def check_plate(self, approx):
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
        plate_height = 150
        plate_image = dst[dst.shape[0] - plate_height:,:]
        # # TODO: remove
        # cv2.imwrite(f"{self.plate_path}/PLT_{self.counter:06d}.jpg", plate_image)
        # self.counter +=1
        # self.make_CNN_chars(dst, plate_image)
        self.slice_plate(dst, plate_image)

    def make_CNN_chars(self, full_image, plate_image):
    
        hsv_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
        hsv_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        mask_full = cv2.inRange(hsv_full, self.full_lower, self.full_upper)
        mask_plate = cv2.inRange(hsv_plate, self.plate_lower, self.plate_upper)
    
        cv2.imshow("cut plate feed", mask_plate)
        cv2.waitKey(3)
        cv2.imshow("full plate feed", mask_full)
        cv2.waitKey(3)

        # self.slice_plate(mask_full, mask_plate)
 
       #TODO: pass both the license plate identifier number image and a
       #list of the different characters on the plate images to
       #read_licence

    def one_hot_map(self, c):
        # Technically could build this array at the start for efficiency
        arr = np.zeros(len(self.ONE_HOT_REF))
        arr[self.ONE_HOT_REF.find(c)] = 1
        return arr

    def hsv_mask(self, img, hsv_lower, hsv_upper):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv, (3,3), cv2.BORDER_DEFAULT)
        return cv2.inRange(blur, hsv_lower, hsv_upper)

    def get_rects(self, img, num_rects = 4, dst = False):
        # Mask -> Get contours -> Take 4 Largest contours
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sorted = np.flip(sorted(contours, key=lambda x: cv2.contourArea(x))[-num_rects:])
        rects = []
        for cnt in cnts_sorted:
            x,y,w,h = cv2.boundingRect(cnt)
            rects.append([x,y,w,h])
        rects = np.array(rects)
        rects = sorted(rects, key=lambda x: x[0])
        if not dst:
            print(rects)
            return rects
        else:
            print(rects)
            return rects[1]

    def get_char(self, img, rect):
        x,y,w,h = rect
        im = cv2.resize(img[y:y+h,x:x+w], self.CHAR_DIMS)
        im = cv2.erode(cv2.dilate(im, self.KERNEL, iterations=1), self.KERNEL, iterations=1)
        print(im.shape)
        return im

    def slice_plate(self, dst, plt):
        dst_mask = self.hsv_mask(dst, self.full_lower, self.full_upper)
        plt_mask = self.hsv_mask(plt, self.plate_lower, self.plate_upper)
        dst_rect = self.get_rects(dst_mask, num_rects=2, dst=True)
        plt_rect = self.get_rects(plt_mask)
        plt_set = [self.get_char(plt_mask, rect) for rect in plt_rect]
        dst_set = self.get_char(dst_mask, dst_rect)
        #TODO: self.read_licence(dst_set, plt_set)
        self.simple_predict(dst_set, plt_set)

    def simple_predict(self, dst, plts):
        identifier = self.ONE_HOT_REF[np.argmax(self.dst_reader.predict(dst))]
        plt_chars = [self.plate_reader.predict(plt) for plt in plts]
        char0 = self.ONE_HOT_REF[np.argmax(plt_chars[0])]
        char1 = self.ONE_HOT_REF[np.argmax(plt_chars[1])]
        char2 = self.ONE_HOT_REF[np.argmax(plt_chars[2])]
        char3 = self.ONE_HOT_REF[np.argmax(plt_chars[3])]
        prediction = str('TeamEthan,notsafe,' + identifier + ',' + char0 + char1 + char2 +char3)
        print(prediction)
        self.licenses.publish(str('TeamEthan,notsafe,' + identifier + ',' + char0
                                                                 + char1
                                                                 + char2
                                                                 +char3))


    # def read_licence(self, plate_identifier_image, plate_char_images):
    #     #TODO: initialize self.submission_timer to 0 in whatever object it belongs to
    #     encoder_len = len(self.ONE_HOT_REF)
    #     def zero_hot(len):
    #         return np.zeros(len, dtype=int)
    #     if time.time() > self.submission_timer + 2:
    #         identity = self.ONE_HOT_REF[np.argmax(self.plate_identifier_guesses)]
    #         char0 = self.ONE_HOT_REF[np.argmax(self.plate_char_guesses[0])]
    #         char1 = self.ONE_HOT_REF[np.argmax(self.plate_char_guesses[1])]
    #         char2 = self.ONE_HOT_REF[np.argmax(self.plate_char_guesses[2])]
    #         char3 = self.ONE_HOT_REF[np.argmax(self.plate_char_guesses[3])]
    #         #This exists in driver.py so this call needs to link to it somehow
    #         self.licenses.publish(str('TeamEthan,notsafe,' + identity + ',' + char0
    #                                                                         + char1
    #                                                                         + char2
    #                                                                         +char3))
    #         self.plate_identifier_guesses = zero_hot(encoder_len)
    #         self.plate_char_guesses = [zero_hot(encoder_len),
    #                                     zero_hot(encoder_len),
    #                                     zero_hot(encoder_len),
    #                                     zero_hot(encoder_len)]
    #     #CNN here is meant to pass the thing in brackets to the CNN and get pack a 36 long 1 hot array of the CNN's guess
    #     plate_identifier_guess = plate_reader.predict(plate_identifier_image)
    #     plate_char_guesses = plate_reader.predict(plate_char_images)
    #     self.plate_identifier_guesses += plate_identifier_guess
    #     for char, i in enumerate(plate_char_guesses):
    #         self.plate_char_guesses[i] += char
    #     self.submission_timer = time.time()

class plate_reader:
    SAVE_PATH = "/home/fizzer/cnn_trainer/letter_model/save/"

    def __init__(self, save_path = SAVE_PATH):
        self.conv_model = models.load_model(save_path)
    
    def predict(self, img):
        print("making prediction")
        img = img.reshape(1, len(img), len(img[0]), -1)
        return self.conv_model.predict(img)

class States(Enum):
    FIND_PLATES = auto()
    WAIT = auto()

print("trying")
rospy.init_node('plate_handler', anonymous = True)
plt = plate_handler()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()