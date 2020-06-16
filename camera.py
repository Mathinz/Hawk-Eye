import cv2, time, pandas
from datetime import datetime
static_back=None
import pandas as pd
import shutil
import os

class VideoCamera(object):

    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None


    def __del__(self):
        self.cap.release()

    def start_motion_detector(self):

        # List when any moving object appear
        motion_list = [None, None]
        # Time of movement
        ttime = []
        # Initializing DataFrame, one column is start
        # time and other column is end time
        df = pd.DataFrame(columns=["Start", "End"])

        global static_back

        img_counter = 0

        # Infinite while loop to treat stack of image as video
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break
            k = cv2.waitKey(1)
            # Initializing motion = 0(no motion)
            motion = 0
            # Converting color image to gray_scale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Converting gray scale image to GaussianBlur
            # so that change can be find easily
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if  static_back is None :
                static_back = gray
                continue

            diff_frame = cv2.absdiff(static_back, gray)

            thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=0)

            (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in cnts:
                if cv2.contourArea(contour) < 10000:
                    continue
                motion = 1
                i = 0
                (x, y, w, h) = cv2.boundingRect(contour)
                # making green rectangle arround the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Appending status of motion
            motion_list.append(motion)

            motion_list = motion_list[-2:]

            # Appending Start time of motion and capture image
            # if number of captured images reach 12 then close the detector
            if  k % 256 == 27 or img_counter == 12:
                # ESC pressed
                print("closing...")
                self.cap.release()
                cv2.destroyAllWindows()
                break
            if motion_list[-1] == 1 and motion_list[-2] == 0:
                i = i+1
                img_name = "static/motion_images/opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
                # ttime.append(datetime.now())
                # print("start time:", i, datetime.now())

            # Appending End time of motion
            if motion_list[-1] == 0 and motion_list[-2] == 1:
                i = i+1
                # ttime.append(datetime.now())
                # print("endtime time:", i, datetime.now())

            cv2.imshow('Motion Detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
