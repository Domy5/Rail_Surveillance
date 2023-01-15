#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Detention of people and/or objects on the railway platform in real time """
# ---------------------------------------------------------------------------

import time
import os
import cv2
import pygame

points_list = []

def measures_time(funcion): # Decorator method to measure times of another method
   def funcion_medida(*args, **kwargs):
       inicio = time.time()
       c = funcion(*args, **kwargs)
       print(time.time() - inicio)
       return c
   return funcion_medida

#@measures_time 
def clear_screen(): # We define the function establishing the name that we want
    """Delete the terminal
        Deletion of the terminal regardless of operating system
    """
    
    if os.name == "posix":
        os.system ("clear")
    elif os.name == "ce" or os.name == "nt" or os.name == "dos":
        os.system ("cls")
    print("")
    print("")
    print("")
    
def is_cuda_cv():
    """Verification CUDA
    1 == using cuda, 0 = not using cuda
    """
    try:  
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return 1
        else:
            return 0
    except:
        return 0
    
def draw_dots(event, x, y, flags, param):
    """mouse event
     We print the information about the events that are being carried out with the mouse
     with the format you need to draw with cv2
    """
    # We print the information about the events that are being carried out with the mouse
       
    global points_list
    points = [', point_11 = ', ', point_12 = ',', point_21 = ', ', point_22 = ',', point_31 = ', ', point_32 = ',', point_41 = ', ', point_42 = ',', point_51 = ', ', point_52 = ']
    list = []

    if event == cv2.EVENT_LBUTTONDOWN:
    # print('event=', event)
    # print('x=', x)
    # print('y=', y)
    # print('flags=', flags)
    # print('[{},{}],'.format(x, y), end="")
        print('{},{},'.format(x, y))
        points_list.append('{}'.format(x))
        points_list.append('{}'.format(y))
        
        if len(points_list) > 9:
            
            str1 = 'ROI_polygon_XX = controler.ROI_polygon(polygon_id = \'XX\''
            
            for p, l in zip(points, points_list):
                list.append("{}\'{}\'".format(p,l))
            
            str2 = ''.join(list)
            str3 =', camera_id = \'XX\')'              
            print (str1 + str2 + str3) 
            points_list = []

def point_in_polygon(point, polygon):
    """Ray casting
    Check if a point is inside a polygon
        
        polygon - List of tuples with the points that form the vertices [(x1, x2), (x2, y2), ..., (xn, yn)]
    """
    i = 0
    x = point[0]
    y = point[1]
    
    j = len(polygon) - 1
    
    output = False
    for i in range(len(polygon)):
        if (polygon[i][1] < y and polygon[j][1] >= y) or (polygon[j][1] < y and polygon[i][1] >= y):
            if polygon[i][0] + (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) * (polygon[j][0] - polygon[i][0]) < x:
                output = not output
        j = i
    return output

def play_track(main_dir, track, play):
    """Play Music in a loop."""
    
    if (play):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.Sound(os.path.join(main_dir, track)).play()
        
        #file = os.path.join(main_dir, track)
        #sonido_fondo = pygame.mixer.Sound(file).play()
        #pygame.mixer.Sound.play(sonido_fondo) # With -1 we indicate that we want it to be repeated indefinitely

def nothing(emp):
    pass