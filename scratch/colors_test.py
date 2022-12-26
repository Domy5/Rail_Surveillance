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

# Colors in BGR

green_color = (0, 255, 0)
red_color = (0, 0, 255)
blue_color = (255, 0, 0)
black_color = (0, 0, 0)
blue_color_cyan = (255, 255, 0)
yellow_color = (0, 255, 255)
white_color = (255, 255, 255)
orange_color = (26, 127, 239)
brown_color = (37, 73, 141)
gray_color = (150, 150, 150)

print("start")

img = np.zeros((900,900,3), np.uint8)

cv2.imshow('image', img)
cv2.circle(img, (450,450), 300, red_color, -1)
cv2.imshow('image', img)
cv2.circle(img, (450,450), 200, yellow_color, -1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

