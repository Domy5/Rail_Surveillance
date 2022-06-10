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
    if event == 1:
       # print('event=', event)
       # print('x=', x)
       # print('y=', y)
       # print('flags=', flags)
       # print('[{},{}],'.format(x, y), end="")
       # print('[{},{}],'.format(x, y))
        lista_puntos.append('[{},{}]'.format(x, y))
        
        if len(lista_puntos) > 4:
            str1 = 'area_pts = np.array(['
            str2 = "'".join(lista_puntos).replace('\'',',')
            str3 ='])'              
            print (str1+str2+str3)  # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])
            
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


def nothing(emp):
    pass