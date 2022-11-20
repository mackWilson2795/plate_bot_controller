#! /usr/bin/env python3

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import time

class comp_driver:

    def __init__(self):
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

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e) #this could get changed to pass so that we don't clog the terminal during comp if there is no image
         
        move = Twist()
        move.linear.x = 0.1
        move.angular.z = 0.8

        cv2.imshow("raw feed", cv_image)
        cv2.waitKey(3)

        try:
            self.mover.publish(move)
        except CvBridgeError as e:
            print(e)

rospy.init_node('comp_driver', anonymous = True)
comp = comp_driver()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()