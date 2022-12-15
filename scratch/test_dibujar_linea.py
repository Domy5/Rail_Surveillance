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

'''En geometría analítica las líneas rectas en un plano pueden ser expresadas mediante una ecuación del tipo y = m x + b, donde x, y son variables en un plano cartesiano. En dicha expresión m  es denominada la "pendiente de la recta" y está relacionada con la  inclinación que toma la recta respecto a un par de ejes que definen el  plano. Mientras que b es el denominado "término independiente" u  "ordenada al origen" y es el valor del punto en el cual la recta corta  al eje vertical en el plano.  
Cita de https://es.wikipedia.org/wiki/Recta'''


VIDEO_PATH = r'videos_pruebas/'
VIDEO = "a1-003 1 minuto 1 via.mkv"

def linea(x0,y0,x1,y1):
    x0 =float(x0)
    x1 =float(x1)
    y0 =float(y0)
    y1 =float(y1)
    
    if((x0==x1) and (y0==y1)):
        print("ES EL MISMO PUNTO")
        return None
    if((x1-x0)!=0):
        m = (y1-y0) / (x1-x0)
        b = m *(-x0) + y0
        print("Y = " + str(m) + "X + (" + str(b) + ")" )
    else:
        print("X = " + str(x0))   

cap = cv2.VideoCapture(f'{VIDEO_PATH}{VIDEO}')
ret, frame = cap.read()

area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])

# Crea una imagen en negro
#img = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)

img = np.zeros((512,512,3), np.uint8)

diferencia_punto1 = (area_pts[2][0] - area_pts[1][0]) // 2
diferencia_punto2 = (area_pts[3][0] - area_pts[4][0]) // 2

p1 = diferencia_punto1 + area_pts[1][0]
p2 = diferencia_punto2 + area_pts[3][0]

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
img = cv2.drawContours(img, [area_pts], -1, (255,0,0), -1)
# pintar linea
#img = cv2.line(img, (34, 0), (0, 255), (150), 4)
img = cv2.line(img, (area_pts[0][0],area_pts[0][1]), (area_pts[3][0],area_pts[3][1]), (0,255 ,0 ), 4)
#img = cv2.line(img, (p1, 0), (0, p2), (0, 0, 255), 1)
img = cv2.line(img, (p1, 0), (area_pts[0][0]+(area_pts[3][0]-area_pts[0][0])//2, area_pts[3][1]+(area_pts[0][1]-area_pts[3][1])//2), (0, 0, 255), 4)
img = cv2.line(img, (p1, 0), (0, round(513.1908713692945)), (0, 100, 0), 1)
#img = cv2.line(img, (p1, 0), (0,1500), (0, 0, 255), 4)

linea(p1, 0, area_pts[0][0]+(area_pts[3][0]-area_pts[0][0])//2, area_pts[3][1]+(area_pts[0][1]-area_pts[3][1])//2)

# Mostrar la imagen
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()