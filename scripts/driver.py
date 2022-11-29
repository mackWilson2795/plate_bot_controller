#! /usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import time

class comp_driver:

    def __init__(self):

        self.state = "outer"
        self.startup_time = time.time() - 1.0

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
        # self.timer = rospy.Subscriber("/clock",
        #                                 ) #What datatype does the clock output?
        time.sleep(1)

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
        MIN_CONTOUR_AREA = 8000

        cut_image = self.raw_cv_image[TOP_CUT:BOTTOM_CUT,:]
        hsv_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2HSV)
        blur_image = cv2.GaussianBlur(hsv_image, (5,5), 0)
        threshold_image = cv2.inRange(blur_image, np.array([0,0,LOWER_THRESHOLD]), np.array([0,0,UPPER_THRESHOLD]))

        new_threshold = threshold_image.copy()

        cv2.imshow("Threshold feed", threshold_image)
        cv2.waitKey(3)

        contours, hierarchy = cv2.findContours(new_threshold.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        max_contour = cntsSorted[-1] #This needs to be changed for thew case when there are no contours on screen
        approx = None
        if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
            epsilon = 0.01*cv2.arcLength(max_contour,True)
            approx = cv2.approxPolyDP(max_contour,epsilon, True)

            new_threshold = cv2.cvtColor(new_threshold, cv2.COLOR_GRAY2BGR)
            
            if len(contours) > 0:
                
                for i in range(len(approx[:,0,1])):
                    # with_approx = cv2.drawContours(new_threshold, approx, -1, (0,0,255), 5)
                    # cv2.imshow("Approx feed", with_approx)
                    # cv2.waitKey(3)
                    approx[i,0,1] = approx[i,0,1] + TOP_CUT

                marked_raw = cv2.drawContours(self.raw_cv_image, approx, -1, (0,0,255), 5)

            else :
                marked_raw = self.raw_cv_image
            
            cv2.imshow("Marked raw feed", marked_raw)
            cv2.waitKey(3)
            # print(approx)
            # print (approx.shape)
            # print(with_approx.shape)

        print(cv2.contourArea(cntsSorted[-1]))
        print("----")

        cv2.imshow("HSV feed", hsv_image)
        cv2.waitKey(3)

        return approx


    def state_machine(self):

        if self.state == "startup":
            move_command = Twist()

            if(time.time() < self.startup_time + 3):
                move_command.linear.x = 0.2
                move_command.angular.z = 0.0
            elif(time.time() < self.startup_time + 6.4):
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

            license_corners = self.seek_license()   

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



rospy.init_node('comp_driver', anonymous = True)
comp = comp_driver()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()

