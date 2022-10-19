#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# https://github.com/Domy5/Rail_Surveillance/
# ---------------------------------------------------------------------------
""" Trabajo fin de Grado, deteccion de objetos en plataforma de vias ferroviarias"""
# ---------------------------------------------------------------------------

# https://blog.roboflow.com/object-detection/
# https://blog.roboflow.com/automl-vs-rekognition-vs-custom-vision/

import argparse
import cv2
import sys
import datetime
import torch
import torchvision
##import torch_tensorrt
import imutils
import time
# from playsound import playsound
import numpy as np
import pandas as pd

import utiles
import controler


loop_flag = 0
pos = 0
numero_fotograma = 0
fpsmax = 0
contador = 0
numero_img = 0
nombre_ventana = 'TFG Domy 0.39'
lista_puntos = []
punto = [0, 0]
contarfoto = 0
cantidad_contornos = 0

color_verde = (0, 255, 0)  # color verde en BGR
color_rojo = (0, 0, 255)  # color rojo en BGR
color_azul = (255, 0, 0)  # color azul en BGR
color_negro = (0, 0, 0)  # color negro en BGR
color_azul_cyan = (255, 255, 0)  # color azul_cyan en BGR
color_amarillo = (0, 255, 255)  # color amarillo en BGR
color_blanco = (255, 255, 255)  # color blanco en BGR
color_naranja = (26, 127, 239)  # color naranja en BGR
color_marron = (37, 73, 141)  # color marron en BGR
color_gris = (150, 150, 150)  # color marron en BGR

uno = True
dos = True
tres = True
cuatro = True
cinco = True
seis = True

utiles.borrarPantalla()

# print(dir(cv2.cuda))

parser = argparse.ArgumentParser(description='Detector de objetos, computación GPU, CPU')

parser.add_argument('-v', '--version', action="store_true",default=False, help='versión del programa')
parser.add_argument('-info', '--informacion', action="store_true",default=False, help='información de las versiones de los paquetes usados')
parser.add_argument('-m', '--mascara', action="store_true",default=True, help='muestra la  mascara')
parser.add_argument('-d', '--deteccion', action="store_true",default=False, help='muestra la detecciones de objetos')
parser.add_argument('-s', '--slicer', action="store_true", default=False,help='muestra barra de desplazamiento (consume muchos recursos)')
parser.add_argument('-mm', '--mouse', action="store_true", default=False,help='muestra por consola las coordenadas de los click')
parser.add_argument('-c', '--procesar_imagen',
                    type=str,
                    choices=['gpu', 'cpu'],
                    default='gpu',  # default='cpu',
                    required=False,
                    help='parámetro GPU o CPU')
parser.add_argument('-i', '--input',
                    type=str,
                    default='1',  # Cambio de camara de video y sus ROI
                    required=False,
                    help='Código administrativo de la estación')

args = parser.parse_args()

if args.version:
    print(nombre_ventana)
    sys.exit("Hasta la vista.")

if args.informacion:
    print(f"Version: {nombre_ventana}")

    utiles.os.system("nvidia-smi")
    print(f"CUDA version: {torch.version.cuda}")
    print("cuda" if torch.cuda.is_available() else "cpu")
    print("1 == using cuda, 0 = not using cuda: ", utiles.is_cuda_cv())
    print("CudaDeviceInfo  version:", cv2.cuda.getDevice())
    print("CudaDeviceInfo  version:", cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice()))
    print("Python version:", sys.version)
    print("OpenCV version:", cv2.__version__)
    print("Numpy  version:", np.__version__)
    print("Torch  version:", torch.__version__)
    print("----", cv2.cuda_DeviceInfo())
    print("Python {}.{}.{}".format(sys.version_info.major,sys.version_info.minor, sys.version_info.micro))
    print("openCV {}".format(cv2.__version__))
    print("openCL activo ? {}".format(cv2.ocl.haveOpenCL()))
    cv2.ocl.setUseOpenCL(True)
    print("openCL {}".format(cv2.ocl.useOpenCL()))

if not torch.cuda.is_available():

    args.procesar_imagen = 'cpu'

    print('No hay GPU habilitada, se usará CPU para el procesado de imagenes')
    print('No hay GPU habilitada, se usará CPU para el procesado de imagenes')
    print('No hay GPU habilitada, se usará CPU para el procesado de imagenes')
    print(' ')

if args.input:

    ruta = controler.get_ruta_video(args.input)
    poligono = controler.get_ROI_video(args.input)

    cap = cv2.VideoCapture(f'{ruta}')
    area_pts = np.array(poligono)
    
if args.procesar_imagen == 'gpu':
    gpu_frame = cv2.cuda_GpuMat()  # para usar GPU

# Model https://pytorch.org/hub/research-models
# Paper https://zenodo.org/record/7002879#.YxNFEHbtb30

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', force_reload=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
# trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
#####    enabled_precisions = {torch.half},
#####    workspace_size = 1 << 22
# )

# print(model.eval())  # https://www.youtube.com/watch?v=3Kae4FF0x0k&t=1s

# (optional list) filter by class, i.e. = [0, 15, 16] for COCO
# 0 = persona, 1 = bicicleta, 6 = tren, 36 = skateboard, 26 = handbag, 16 = dog, 24  backpack, 13    bench, 28  suitcase
model.classes = [0, 6]
model.conf = 0.3  # NMS confidence threshold   
model.iou = 0.45  # NMS IoU threshold (0-1)  https://kikaben.com/object-detection-non-maximum-suppression/
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

cv2.namedWindow(nombre_ventana)
cv2.moveWindow(nombre_ventana, 10, 50)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cap.get(cv2.CAP_PROP_FPS))

seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("#######################")
print("## Ejecución en: {} ##".format(args.procesar_imagen))
print("#######################")
print("Duracion en segundos :", seconds)
print("Tiempo de video      :", video_time)

print("HEIGHT: {}".format(height))
print("WIDTH: {}".format(width))
print("")
print("OSD:")
print('1:-> Infor Alarma, FPS, N fotos')
print('2:-> ROI')
print('3:-> Contornos dentro de ROI por subtracion de fondo')
print('4:-> Punto pie derecho')
print('5:-> Contornos en la escena (Personas, trenes, bolsos, carros)')
print('6:-> Activar mejor rendimiento')
       

# utiles.borrarPantalla()

if args.slicer:
    cv2.createTrackbar('time', nombre_ventana, 0, frames, utiles.nothing)
    
if args.mouse:
    cv2.setMouseCallback(nombre_ventana, utiles.dibujando)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG( )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200,nmixtures = 5, backgroundRatio = 0.7,noiseSigma = 0 )
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg = cv2.createBackgroundSubtractorMOG2()

Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:

    start = time.time()

    if args.slicer:
        if loop_flag == pos:
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', nombre_ventana, loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', nombre_ventana)
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    ret, frame = cap.read()
    
    if ret == False:
        break
    
    contorno = False
    tren = False
    persona_en_via = False

    ## ## # Mejoramos rendimiento
    ## if seis:
    ##     contarfoto += 1
    ##     if contarfoto % 2 !=0:
    ##         print("Mejora rendimoento {}....{}".format(seis, contarfoto))
    ##         continue

    if args.procesar_imagen == 'gpu':
        gpu_frame.upload(frame)  # para usar GPU
        gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)  # para usar GPU
        gray = gray.download()  # para usar GPU
    elif args.procesar_imagen == 'cpu':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("error de procesador de imagenes")

    # Detección
    #detect = model(frame, size = 640)
    detect = model(frame)
    
    #detect.print()
        #Speed: 11.0ms pre-process, 59.3ms inference, 3.0ms NMS per image at shape (1, 3, 512, 640)
        #image 1/1: 480x640 5 persons

    # info es un objeto de tipo <class 'pandas.core.frame.DataFrame'>
    info = detect.pandas().xyxy[0]

    # .loc y .iloc . La diferencia entre ellos es que .loc acepta etiquetas y .iloc – índices. También cuando usamos los accesores primero especificamos filas y luego columnas.

    # print(info.shape[0]) #numero de filas

    for i in range(info.shape[0]):
        
        x0 = info.iat[i, 0]
        y0 = info.iat[i, 1]
        x1 = info.iat[i, 2]
        y1 = info.iat[i, 3]
        confidence = info.iat[i, 4]
        clase = info.iat[i, 5]
        name = info.iat[i, 6]

        # print("Clase: {}, Name: {}, Confiabilidad {}.".format(clase, name, confidence))

        if clase == 0:  # Persona
            
            # pintamos el un punto en esquina inferior derecha de cada persona detectata
            x = x1.astype(int)
            y = y1.astype(int)
            punto = [x - 10, y - 10]
            if cuatro:
                # Correción Tutor 28/junio
                cv2.circle(frame, tuple(punto), 1, color_azul_cyan, 3)

        if clase == 6:  # Tren
            tren = True
            
        if cinco:
            cv2.imshow(nombre_ventana, np.squeeze(detect.render()))

    if utiles.punto_en_poligono(punto, area_pts):
        persona_en_via = True
        #playsound(AUDIO_ARCHIVO, False)

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    imagen_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = fgbg.apply(imagen_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, Kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in cnts:
        if cv2.contourArea(cnt) > 250:  # 500
            x, y, w, h = cv2.boundingRect(cnt)
            if tres:
                # dibuja los contornos que se encuentran
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_amarillo, 2)
                contorno = True
    
###############################
    
    end = time.time()

    fps = 1 / (end - start)
    fps1 = cap.get(cv2.CAP_PROP_FPS)

    # Obtener el número de marcos de video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numero_fotograma = numero_fotograma + 1

    # Para que de tiempo a ver los fps
    if contador < 5:
        contador = contador + 1
    else:
        fpsmax = fps
        contador = 0

###############################

#  logica .. si hay tren.. si hay personas en la via, si ha caido un objeto

    texto_estado = 'Sin Moviemiento'
    color = color_verde

    if contorno :
        cantidad_contornos +=1
        
        if cantidad_contornos == 5:
            texto_estado = 'Precaucion Movimiento'
            color = color_amarillo
            cantidad_contornos = 0
        
    if persona_en_via :
        texto_estado = 'ALERTA Movimiento'
        color = color_rojo
        print("PERSONAS EN LA VIA¡¡¡")
        
    if  tren and persona_en_via :
        texto_estado = 'ALERTA Movimiento'
        color = color_rojo
        print("LLEGADA DEL TREN CON PERSONA EN LA VIA¡¡¡¡¡")
            
    if  tren : # QUITAR CONTORNOS SI ESTA EL TREN 
        tres = False
    else:
        tres = True

###############################

    if uno:
      
       cv2.putText(frame, texto_estado, (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       cv2.putText(frame, "FPS: {:.2f}".format(fpsmax), (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       cv2.putText(frame, "N fotos: {:.2f}".format(numero_fotograma), (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)      
       
       #cv2.putText(frame, texto_estado, (height -130, width -240),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       #cv2.putText(frame, "FPS: {:.2f}".format(fpsmax), (height -130, width -210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       #cv2.putText(frame, "N fotos: {:.2f}".format(numero_fotograma), (height -130, width -180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.putText(frame, "FPS?: {:.2f}".format(fps1), (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.putText(frame, "Total fotogramas: {:.2f}".format(total_frames), (10, 160),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if dos:
        cv2.drawContours(frame, [area_pts], -1, color, 2)  # dibuja ROI
        
    if args.mascara:
        cv2.imshow('fgmask', fgmask)
        #cv2.moveWindow('fgmask', 10, 75 + height);

    if args.deteccion:
        cv2.imshow(nombre_ventana, np.squeeze(detect.render()))
        print(detect.render())
        
    # Mostramos FPS
    cv2.imshow(nombre_ventana, frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # Cerrar ventana desde Esc
        break

    if k == ord('p'):  # parar si presionas p
        cv2.waitKey(-1)

    if k == ord('c'):  # captura de pantalla
        print('C:\\capturas\\' + str(numero_img) + '.jpg')

        if not cv2.imwrite('C:\\capturas\\' + str(numero_img) + '.jpg', frame):
            raise Exception("No se pudo escribir la imagen")
        numero_img += 1
        
# OSD On Screen Display
    if k == ord('1'):                       # Infor Alarma, FPS, N fotos
        uno = False if uno else True
    if k == ord('2'):                       # ROI
        dos = False if dos else True   
    if k == ord('3'):                       # Contornos dentro de ROI por subtracion de fondo
        tres = False if tres else True      
    if k == ord('4'):                       # Punto pie derecho  
        cuatro =  False if cuatro else True   
    if k == ord('5'):                       # Contornos en la escena (Personas, trenes, bolsos, carros)
        cinco = False if cinco else True    
    if k == ord('6'):                       # Activar mejor rendimiento
        seis = False if seis else True

    # Cerrar ventana desde X
    elif cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_AUTOSIZE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
