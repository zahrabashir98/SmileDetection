# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2
from predict import predict, draw_face_info, show_image


def run():
    cv2.namedWindow("Webcam")
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return

    while True:
        rval, frame = capture.read()
        if not rval:
            break

        image = frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for face_info in predict(gray_image):
            draw_face_info(image, face_info)

        cv2.imshow("Webcam", image)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'): # exit on ESC or Q
            break

    cv2.destroyWindow("Webcam")
    capture.release()



if __name__ == '__main__':
    run()
