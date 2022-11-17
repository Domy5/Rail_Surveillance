import argparse
import cv2
import sys
import datetime
import torch
import time
import numpy as np
import pandas as pd
import utiles

color_verde = (0, 255, 0)  # color verde en BGR
color_rojo = (0, 0, 255)  # color rojo en BGR
color_azul = (255, 0, 0)  # color azul en BGR
color_negro = (0, 0, 0)  # color negro en BGR
color_azul_cyan = (255, 255, 0)  # color azul_cyan en BGR
color_amarillo = (0, 255, 255)  # color azul_cyan en BGR
color_blanco = (255, 255, 255)  # color azul_cyan en BGR
color_naranja = (26, 127, 239)  # color azul_cyan en BGR
color_marron = (37, 73, 141)  # color azul_cyan en BGR


print("empeznado")

img = np.zeros((900,900,3), np.uint8)

cv2.imshow('image', img)
cv2.circle(img, (450,450), 300, color_azul, -1)
cv2.imshow('image', img)
cv2.circle(img, (450,450), 200, color_naranja, -1)

uno = False


uno = uno if False else True

print(uno)

# Mostrar la imagen
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

