#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Domingo Martínez Núñez
# Created Date: abril 2022
# version ='1.0'
# https://github.com/Domy5/Rail_Surveillance/
# ---------------------------------------------------------------------------
""" Deteccion de objetos en plataforma de vias ferroviarias"""
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
from playsound import playsound
import numpy as np
import pandas as pd

import tools.utils as utils
import controler


loop_flag = 0
pos = 0
frame_number = 0
fpsmax = 0
counter = 0
img_number = 0
windows_name = 'Rail_Surveillance 0.42'
points_list = []
point = [0, 0]
frame_counter = 0
contour_quantity = 0

green_color = (0, 255, 0)  # color verde en BGR
red_color = (0, 0, 255)  # color rojo en BGR
blue_color = (255, 0, 0)  # color azul en BGR
black_color = (0, 0, 0)  # color negro en BGR
blue_color_cyan = (255, 255, 0)  # color azul_cyan en BGR
yellow_color = (0, 255, 255)  # color amarillo en BGR
white_color = (255, 255, 255)  # color blanco en BGR
orange_color = (26, 127, 239)  # color naranja en BGR
brown_color = (37, 73, 141)  # color marron en BGR
gray_color = (150, 150, 150)  # color marron en BGR

flag_1 = True
flag_2 = True
flag_3 = True
flag_4 = True
flag_5 = True
flag_6 = True
flag_7 = True

utils.clear_screen()

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
    print(windows_name)
    sys.exit("Hasta la vista.")

if args.informacion:
    print(f"Version: {windows_name}")

    utils.os.system("nvidia-smi")
    print(f"CUDA version: {torch.version.cuda}")
    print("cuda" if torch.cuda.is_available() else "cpu")
    print("1 == using cuda, 0 = not using cuda: ", utils.is_cuda_cv())
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
    print(' ')

if args.input:

    route = controler.get_ruta_video(args.input)
    polygon = controler.get_ROI_video(args.input)

    cap = cv2.VideoCapture(f'{route}')
    area_pts = np.array(polygon)
    
if args.procesar_imagen == 'gpu':
    gpu_frame = cv2.cuda_GpuMat()  # para usar GPU

# Model https://pytorch.org/hub/research-models
# Paper https://zenodo.org/record/7002879#.YxNFEHbtb30

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', force_reload=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

#model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True, force_reload=True)


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

cv2.namedWindow(windows_name)
cv2.moveWindow(windows_name, 10, 50)

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
print('Opciones:')
print('------------------------------')
print('- Esc :-> Cierra la ejecución del video')
print('- p   :-> Parar el video')
print('- c   :-> Captura un frame del video y lo guarda en \\capturas\\numero_img.jpg')
print('- s   :-> Activar sonoria de alarmas')
print('')
print('------------------------------')
print('')
print('OSD:')
print('------------------------------')
print('- 1 :-> Infor Alarma, FPS, N fotos')
print('- 2 :-> ROI')
print('- 3 :-> Contornos dentro de ROI por subtracion de fondo')
print('- 4 :-> Punto pie derecho en personas')
print('- 5 :-> Contornos en la escena (Personas, trenes, bolsos, carros)')
print('- 6 :-> Activar mejor rendimiento')
print('------------------------------')
print('')
       

# utiles.clear_screen()

if args.slicer:
    cv2.createTrackbar('time', windows_name, 0, frames, utils.nothing)
    
if args.mouse:
    cv2.setMouseCallback(windows_name, utils.draw_dots)

# Subtractors
#mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
#mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, False)
#gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
#knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
#cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG( )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200,nmixtures = 5, backgroundRatio = 0.7,noiseSigma = 0 )
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()

mog2Subtractor = cv2.createBackgroundSubtractorMOG2()

# https://docs.opencv.org/4.x/d2/d55/group__bgsegm.html


Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:

    start = time.time()

    if args.slicer:
        if loop_flag == pos:
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', windows_name, loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', windows_name)
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    ret, frame = cap.read()
    
    if not ret:
        break
    
    contorno = False
    tren = False
    persona_en_via = False

    ## ## # Mejoramos rendimiento
    ## if flag_6:
    ##     frame_counter += 1
    ##     if frame_counter % 2 !=0:
    ##         print("Mejora rendimoento {}....{}".format(flag_6, frame_counter))
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
            
            # pintamos el un point en esquina inferior derecha de cada persona detectata
            x = x1.astype(int)
            y = y1.astype(int)
            point = [x - 10, y - 10]
            if flag_4:
                # Correción Tutor 28/junio
                cv2.circle(frame, tuple(point), 1, blue_color_cyan, 3)

        if clase == 6:  # Tren
            tren = True
            
        if flag_5:
            cv2.imshow(windows_name, np.squeeze(detect.render()))

    if utils.point_in_polygon(point, area_pts):
        persona_en_via = True
        #playsound(AUDIO_ARCHIVO, False)

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    imagen_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = mog2Subtractor.apply(imagen_area)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, Kernel)  # Sería mejor aplicar apertura morfológica al resultado para eliminar los ruidos. // https://docs.opencv.org/4.x/d8/d38/tutorial_bgsegm_bg_subtraction.html
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in cnts:
        if cv2.contourArea(cnt) > 250:  # 500
            x, y, w, h = cv2.boundingRect(cnt)
            if flag_3:
                # dibuja los contornos que se encuentran
                cv2.rectangle(frame, (x, y), (x + w, y + h), yellow_color, 2)
                contorno = True
    
###############################
    
    end = time.time()

    fps = 1 / (end - start)
    fps1 = cap.get(cv2.CAP_PROP_FPS)

    # Obtener el número de marcos de video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = frame_number + 1

    # Para que de tiempo a ver los fps
    if counter < 5:
        counter = counter + 1
    else:
        fpsmax = fps
        counter = 0

###############################

#  logica .. si hay tren.. si hay personas en la via, si ha caido un objeto

    texto_estado = 'Motionless'
    color = green_color

    if contorno :
        contour_quantity +=1
        
        if contour_quantity == 5:
            texto_estado = 'Caution Movement'
            color = yellow_color
            contour_quantity = 0
            utils.play_track('assets/sound', 'alarm_2.wav', flag_7)
        
    if persona_en_via :
        texto_estado = 'ALERT Movement'
        color = red_color
        print("PEOPLE ON THE TRAIN TRACK¡¡")
        utils.play_track('assets/sound', 'alarm_1.wav', flag_7)
        
    if  tren and persona_en_via :
        texto_estado = 'ALERT Movement'
        color = red_color
        print("ARRIVAL OF THE TRAIN WITH PERSON ON THE TRAIN TRACK¡¡")
            
    if  tren : # QUITAR CONTORNOS SI ESTA EL TREN 
        flag_3 = False
    else:
        flag_3 = True

###############################

    if flag_1:
      
       cv2.putText(frame, texto_estado, (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
      # cv2.putText(frame, 'FPS: {:.2f}'.format(fpsmax), (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
      # cv2.putText(frame, 'N fotos: {:.2f}'.format(frame_number), (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)      
      # cv2.putText(frame, 'Alert sound:{}'.format(flag_7), (10, 105),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)      
       
       #cv2.putText(frame, texto_estado, (height -130, width -240),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       #cv2.putText(frame, "FPS: {:.2f}".format(fpsmax), (height -130, width -210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       #cv2.putText(frame, "N fotos: {:.2f}".format(frame_number), (height -130, width -180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.putText(frame, "FPS?: {:.2f}".format(fps1), (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.putText(frame, "Total fotogramas: {:.2f}".format(total_frames), (10, 160),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if flag_2:
        cv2.drawContours(frame, [area_pts], -1, color, 2)  # dibuja ROI
        cv2.drawContours(fgmask, [area_pts], -1, gray_color, 2) # dibuja ROI en Background Subtraction in ROI
        
    if args.mascara:
        cv2.imshow('Background Subtraction in ROI', fgmask)
        #cv2.moveWindow('fgmask', 10, 75 + height);

    if args.deteccion:
        cv2.imshow(windows_name, np.squeeze(detect.render()))
        print(detect.render())
        
    # Mostramos FPS
    cv2.imshow(windows_name, frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # Cerrar ventana desde Esc
        break

    if k == ord('p'):  # parar si presionas p
        cv2.waitKey(-1)

    if k == ord('c'):  # captura de pantalla
        print('C:\\capturas\\' + str(img_number) + '.jpg')

        if not cv2.imwrite('C:\\capturas\\' + str(img_number) + '.jpg', frame):
            raise Exception("No se pudo escribir la imagen")
        img_number += 1
        
# OSD On Screen Display
    if k == ord('1'):                       # Infor Alarma, FPS, N fotos
        flag_1 = False if flag_1 else True
    if k == ord('2'):                       # ROI
        flag_2 = False if flag_2 else True   
    if k == ord('3'):                       # Contornos dentro de ROI por subtracion de fondo
        flag_3 = False if flag_3 else True      
    if k == ord('4'):                       # Punto pie derecho  
        flag_4 =  False if flag_4 else True   
    if k == ord('5'):                       # Contornos en la escena (Personas, trenes, bolsos, carros)
        flag_5 = False if flag_5 else True    
    if k == ord('6'):                       # Activar mejor rendimiento
        flag_6 = False if flag_6 else True    
    if k == ord('s'):                       # Activar sonoria de alarmas
        flag_7 = False if flag_7 else True

    # Cerrar ventana desde X
    elif cv2.getWindowProperty(windows_name, cv2.WND_PROP_AUTOSIZE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
