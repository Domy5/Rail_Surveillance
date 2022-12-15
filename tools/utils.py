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

lista_puntos = []

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
       
    global lista_puntos
    puntos = [', punto11 = ', ', punto12 = ',', punto21 = ', ', punto22 = ',', punto31 = ', ', punto32 = ',', punto41 = ', ', punto42 = ',', punto51 = ', ', punto52 = ']
    lista = []
    
    if event == 1:
    # print('event=', event)
    # print('x=', x)
    # print('y=', y)
    # print('flags=', flags)
    # print('[{},{}],'.format(x, y), end="")
        print('{},{},'.format(x, y))
        lista_puntos.append('{}'.format(x))
        lista_puntos.append('{}'.format(y))
        
        if len(lista_puntos) > 9:
            
            str1 = 'ROI_poligonoXXX = ROI_poligono(id_poligono = \'XXX\''
            #str2 = "'".join(lista_puntos).replace('\'',',')
            
            for p, l in zip(puntos, lista_puntos):
                lista.append("{}\'{}\'".format(p,l))
            
            str2 = ''.join(lista)
            str3 =', id_camara = \'XXX\')'              
            print (str1 + str2 + str3) # ROI_poligono2 = ROI_poligono(id_poligono = '2', punto11 = '1', punto12 = '1', punto21 = '1', punto22 = '1', punto31 = '1', punto32 = '1', punto41 = '1', punto42 = '1', punto51 = '1', punto52 = '1', id_camara = '2') 
  
            lista_puntos = []

def point_in_polygon(punto, poligono):
    """Ray casting
    Check if a point is inside a polygon
        
        polygon - List of tuples with the points that form the vertices [(x1, x2), (x2, y2), ..., (xn, yn)]
    """
    i = 0
    x = punto[0]
    y = punto[1]
    
    j = len(poligono) - 1
    
    salida = False
    for i in range(len(poligono)):
        if (poligono[i][1] < y and poligono[j][1] >= y) or (poligono[j][1] < y and poligono[i][1] >= y):
            if poligono[i][0] + (y - poligono[i][1]) / (poligono[j][1] - poligono[i][1]) * (poligono[j][0] - poligono[i][0]) < x:
                salida = not salida
        j = i
    return salida

def play_track(main_dir, track, play):
    """Play Music in a loop."""
    
    if (play):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.Sound(os.path.join(main_dir, track)).play()
        
        #file = os.path.join(main_dir, track)
        #sonido_fondo = pygame.mixer.Sound(file).play()
        #pygame.mixer.Sound.play(sonido_fondo) # Con -1 indicamos que queremos que se repita indefinidamente

def nothing(emp):
    pass