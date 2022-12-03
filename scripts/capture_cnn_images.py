#! /usr/bin/env python3
import sys
import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class image_producer:
    def __init__(self):
        self.im_path = '/home/fizzer/CNN_images/'
        self.vel_state = [0.,0.]
        self.vel_last = [0.,0.]
        self.bridge = CvBridge()
        self.im_num = self.set_im_num()
        self.max_images = 1000000
        self.frame_count = 0
        self.frame_skip = 2
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.img_callback)
        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.vel_callback)
    
    def img_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Drive view", cv_image)
        cv2.waitKey(1)
        if (self.im_num < self.max_images and self.vel_state != [0.0,0.0] and
                            (self.frame_count >= self.frame_skip or self.vel_state != self.vel_last)):
            print(f"Writing img to file: {self.im_path}{self.im_num:06d}")
            cv2.imwrite(f"{self.im_path}{self.vel_state[0]}_{self.vel_state[1]}_{self.im_num:06d}.jpg", cv_image)     
            self.im_num += 1
            self.frame_count = 0
            if self.vel_state != [0.0,0.0]:
                self.vel_last = self.vel_state
        self.frame_count += 1
        
    def set_im_num(self):
        file_list = [s[-10:-4] for s in os.listdir(self.im_path)]
        list.sort(file_list)
        if (len(file_list) > 0):
            print(f"Previously saved images detected - starting from image {int(file_list[-1])}")
            return int(file_list[-1])
        else:
            return 0

    def vel_callback(self,data):
        data = str(data).split(':')
        x_vel = float(data[2].split()[0])
        z_ang = float(data[-1].split()[0])
        self.vel_state = [x_vel, z_ang]
        print(self.vel_state)
        

def main(args):
  ip = image_producer()
  rospy.init_node('image_producer', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)