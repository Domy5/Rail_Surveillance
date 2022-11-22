#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Trabajo fin de Grado, deteccion de objetos en plataforma de vias ferroviarias"""
# ---------------------------------------------------------------------------

import time
import os
import cv2
import pygame

lista_puntos = []

def mide_tiempo(funcion): # Metodo decorador para medir tiempos de otro metodo
   def funcion_medida(*args, **kwargs):
       inicio = time.time()
       c = funcion(*args, **kwargs)
       print(time.time() - inicio)
       return c
   return funcion_medida

#@mide_tiempo 
def borrarPantalla(): #Definimos la función estableciendo el nombre que queramos
    """Borra la terminar
    Borrado de la terminal independientemente de sistema operativo
    """
    
    if os.name == "posix":
        os.system ("clear")
    elif os.name == "ce" or os.name == "nt" or os.name == "dos":
        os.system ("cls")
    print("")
    print("")
    print("")
    
def is_cuda_cv():
    """Comprobación CUDA
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
    
def dibujando(event, x, y, flags, param):
    """Evento ratón
    Imprimimos la información sobre los eventos que se estén realizando con el raton 
        con el formato que necesita para dibujar con cv2
    """
    # Imprimimos la información sobre los eventos que se estén realizando con el raton
    
       
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

def punto_en_poligono(punto, poligono):
    """Ray casting
    Comprueba si un punto se encuentra dentro de un polígono
        
       poligono - Lista de tuplas con los puntos que forman los vértices [(x1, x2), (x2, y2), ..., (xn, yn)]
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

def play_track(track_path):
    """Play Music in a loop."""
    
    pygame.init()
    pygame.mixer.init()
    
    sonido_fondo = pygame.mixer.Sound(track_path)
    pygame.mixer.Sound.play(sonido_fondo) # Con -1 indicamos que queremos que se repita indefinidamente



def nothing(emp):
    pass