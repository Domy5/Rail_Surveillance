#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# https://github.com/Domy5/Rail_Surveillance/
# ---------------------------------------------------------------------------
""" Detention of people and/or objects on the railway platform in real time """
# ---------------------------------------------------------------------------

import argparse
import cv2
import sys
import datetime
import torch
import time
import numpy as np
import pandas as pd

'''In analytic geometry, straight lines in a plane can be expressed by an equation of the type y = m x + b, where x, y are variables in a Cartesian plane. In said expression m is called the "slope of the line" and is related to the inclination that the line takes with respect to a pair of axes that define the plane. While b is the so-called "independent term" or "ordinate to the origin" and is the value of the point at which the line intersects the vertical axis in the plane.
Quote from https://es.wikipedia.org/wiki/Recta'''


VIDEO_PATH = r'video_test/'
VIDEO = "a1-003 1 minuto 1 via.mkv"

def line(x0,y0,x1,y1):
    x0 =float(x0)
    x1 =float(x1)
    y0 =float(y0)
    y1 =float(y1)
    
    if((x0==x1) and (y0==y1)):
        print("IT'S THE SAME POINT")
        return None
    if((x1-x0)!=0):
        m = (y1-y0) / (x1-x0)
        b = m *(-x0) + y0
        print("Y = " + str(m) + "X + (" + str(b) + ")" )
    else:
        print("X = " + str(x0))   

cap = cv2.VideoCapture(f'{VIDEO_PATH}{VIDEO}')
ret, frame = cap.read()

point_area = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])

# Create a black image
# img = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)

img = np.zeros((512,512,3), np.uint8)

difference_point_1 = (point_area[2][0] - point_area[1][0]) // 2
difference_point_2 = (point_area[3][0] - point_area[4][0]) // 2

p1 = difference_point_1 + point_area[1][0]
p2 = difference_point_2 + point_area[3][0]

# print(punto1)
# print(punto2)
#
#   y
#    |
#    |        p1 .
#    |
#    |
#    |
#    |  p2 .
#    ---------------------- x

#  (x,y)   

# Pinta poligono
img = cv2.drawContours(img, [point_area], -1, (255,0,0), -1)
# pintar linea
#img = cv2.line(img, (34, 0), (0, 255), (150), 4)
img = cv2.line(img, (point_area[0][0],point_area[0][1]), (point_area[3][0],point_area[3][1]), (0,255 ,0 ), 4)
#img = cv2.line(img, (p1, 0), (0, p2), (0, 0, 255), 1)
img = cv2.line(img, (p1, 0), (point_area[0][0]+(point_area[3][0]-point_area[0][0])//2, point_area[3][1]+(point_area[0][1]-point_area[3][1])//2), (0, 0, 255), 4)
img = cv2.line(img, (p1, 0), (0, round(513.1908713692945)), (0, 100, 0), 1)
#img = cv2.line(img, (p1, 0), (0,1500), (0, 0, 255), 4)

line(p1, 0, point_area[0][0]+(point_area[3][0]-point_area[0][0])//2, point_area[3][1]+(point_area[0][1]-point_area[3][1])//2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()