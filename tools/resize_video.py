#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Deteccion de objetos en plataforma de vias ferroviarias"""
# ---------------------------------------------------------------------------

import cv2
import tools.utils as utils

VIDEO = "a1-003 1 minuto 1 via.mkv"
VIDEO_PATH = r'test_video/'

HEIGHT = 480
WIDTH = 640

#cap = cv2.VideoCapture('C:/New folder/video.avi')
cap = cv2.VideoCapture(VIDEO_PATH+VIDEO)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

output = VIDEO_PATH + VIDEO.replace("mkv", "avi")
count = 0

print(output)

out = cv2.VideoWriter(output ,fourcc, 5, (WIDTH,HEIGHT))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(WIDTH,HEIGHT),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break
    count+=1
    utils.borrarPantalla()
    print(count)
    
cap.release()
out.release()
cv2.destroyAllWindows()