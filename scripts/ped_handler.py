import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from enum import Enum, auto

class ped_handler:
    HSV_THRESH = {
        "uh": 5,
        "us": 255,
        "uv": 255,
        "lh": 0,
        "ls": 250,
        "lv": 250
    }


    
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",
                                            Image,
                                            self.img_callback)
        self.lower_hsv = np.array([self.HSV_THRESH["lh"], self.HSV_THRESH["ls"],self.HSV_THRESH["lv"]])
        self.upper_hsv = np.array([self.HSV_THRESH["uh"], self.HSV_THRESH["us"],self.HSV_THRESH["uv"]])
        self.state = States.FIND_LINE
    
    def img_callback(self, data):
        MIN_CNT_AREA = 20000
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        if self.state == States.FIND_LINE or self.state == States.PREP_STOP: 
            hsv = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), 7)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            contours, hierarchy = cv2.findContours(mask, 
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            if len(cntsSorted) > 0 and cv2.contourArea(cntsSorted[-1]) > MIN_CNT_AREA:
                print(cv2.contourArea(cntsSorted[-1]))
                cv2.imshow("Red Mask", mask)
                cv2.waitKey(1)
                self.state = States.TRACK_PED
            elif self.state == States.PREP_STOP:
                # TODO: Publish stop command
                self.state = States.TRACK_PED
        if self.state == States.TRACK_PED:
            self.state = States.FIND_LINE

class States(Enum):
    FIND_LINE = auto()
    PREP_STOP = auto()
    TRACK_PED = auto()
    DRIVE = auto()
        
rospy.init_node('ped_handler', anonymous = True)
ped = ped_handler()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting Down")
cv2.destroyAllWindows()
