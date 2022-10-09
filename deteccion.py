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
import time
# from playsound import playsound
import numpy as np
import pandas as pd

import utiles, controler

loop_flag = 0
pos = 0
numero_fotograma = 0
fpsmax = 0
contador = 0
numero_img = 0
scale = 0.5
nombre_ventana = 'TFG Domy 0.26'
lista_puntos = []
contafoto = 0

punto = [0, 0]

OSD = 6  # OSD significa "On Screen Display"

utiles.borrarPantalla()

# print(dir(cv2.cuda))

parser = argparse.ArgumentParser(
    description='Detector de objetos, computación GPU, CPU')

parser.add_argument('-v', '--version', action="store_true",
                    default=False, help='versión del programa')
parser.add_argument('-info', '--informacion', action="store_true",
                    default=False, help='información de las versiones de los paquetes usados')
parser.add_argument('-m', '--mascara', action="store_true",
                    default=False, help='muestra la  mascara')
parser.add_argument('-d', '--deteccion', action="store_true",
                    default=False, help='muestra la detecciones de objetos')
parser.add_argument('-s', '--slicer', action="store_true", default=False,
                    help='muestra barra de desplazamiento (consume muchos recursos)')
parser.add_argument('-mm', '--mouse', action="store_true", default=False,
                    help='muestra por consola las coordenadas de los click')

parser.add_argument('-c', '--procesar_imagen',
                    type=str,
                    choices=['gpu', 'cpu'],
                    default='gpu',  # default='cpu',
                    required=False,
                    help='parámetro GPU o CPU')
parser.add_argument('-i', '--input',
                    type=str,
                    default='1', # Cambio de camara de video
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
    #print("cuda numba: ",cuda.gpus)
    print("1 == using cuda, 0 = not using cuda: ", utiles.is_cuda_cv())
    print("CudaDeviceInfo  version:", cv2.cuda.getDevice())
    print("CudaDeviceInfo  version:",
          cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice()))
    print("Python version:", sys.version)
    print("OpenCV version:", cv2.__version__)
    #print("Numba  version:", numba.__version__)
    print("Numpy  version:", np.__version__)
    print("Torch  version:", torch.__version__)
    print("----", cv2.cuda_DeviceInfo())
    print("Python {}.{}.{}".format(sys.version_info.major,
                                   sys.version_info.minor, sys.version_info.micro))
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

#VIDEO_PATH = r'videos_pruebas/'
#AUDIO_ARCHIVO = './' + 'Alarma.mp3'
#VIDEO = "a1-003 1 minuto 1 via.mkv"


if args.input:

    ruta = controler.get_ruta_video(args.input)
    poligono = controler.get_ROI_video(args.input)
    
    cap = cv2.VideoCapture(f'{ruta}')
    area_pts = np.array(poligono)

#VIDEO = "output.avi"
#VIDEO = "Madrid_un_policier_sauve_une_femme_tombe_sur_les_v.mp4"

# Model https://pytorch.org/hub/research-models
# model = torch.hub.load('ultralytics/yolov5', 'YOLOv5x-cls', force_reload=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
##model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')


# model = torch.hub.load('ultralytics/yolov5', 'yolov5s-cls')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelos/yolov5s.pt') # cuidad con cada plataforma
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelos/yolov5s-cls.pt') # ver esto --> https://zenodo.org/record/7002879#.YxNFEHbtb30

# (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
# 0 = persona, 1 = bicicleta, 6 = tren, 36 = skateboard, 26 = handbag, 16 = dog, 24  backpack, 13    bench, 28  suitcase
model.classes = [0, 6]
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold (0-1)
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image
# model.amp = False  # Automatic Mixed Precision (AMP) inference



# cap = cv2.VideoCapture(VIDEO_PATH,cv2.CAP_INTEL_MFX)

# gpu_frame = cv2.cuda_GpuMat(n,m,cv2.CV_8UC4)

if args.procesar_imagen == 'gpu':
    gpu_frame = cv2.cuda_GpuMat()  # para usar GPU

cv2.namedWindow(nombre_ventana)
cv2.moveWindow(nombre_ventana, 10, 50)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cap.get(cv2.CAP_PROP_FPS))

seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))

print("#######################")
print("## Ejecución en: {} ##".format(args.procesar_imagen))
print("#######################")
print("Duracion en segundos :", seconds)
print("Tiempo de video      :", video_time)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("HEIGHT: {}".format(height))
print("WIDTH: {}".format(width))

# utiles.borrarPantalla()

if args.slicer:
    cv2.createTrackbar('time', nombre_ventana, 0, frames, utiles.nothing)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG( )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200,nmixtures = 5, backgroundRatio = 0.7,noiseSigma = 0 )
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg = cv2.createBackgroundSubtractorMOG2()

Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

if args.mouse:
    cv2.setMouseCallback(nombre_ventana, utiles.dibujando)

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

    # Mejoranos rendieniento
    #contafoto += 1
    # if contafoto % 3 !=0:
    #    continue

    if args.procesar_imagen == 'gpu':
        gpu_frame.upload(frame)  # para usar GPU
        gray = cv2.cuda.cvtColor(
            gpu_frame, cv2.COLOR_BGR2GRAY)  # para usar GPU
        gray = gray.download()  # para usar GPU
        #resized = cv2.cuda.resize(gpu_frame, (int(1280 * scale), int(720 * scale)))
    elif args.procesar_imagen == 'cpu':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("error de procesador de imagenes")

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), 1)

    #punto1Rectangulo = (30,400)
    #punto2Rectangulo = (300,450)
    #color = (0,0,0)
    #
    #negro = cv2.rectangle(frame, punto1Rectangulo, punto2Rectangulo, color, cv2.FILLED)

    color = (0, 255, 0)
    texto_estado = "Sin Movimiento"

    # deteccion
    # detect = model(frame, size = 640)
    detect = model(frame)

    # info es un objeto de tipo <class 'pandas.core.frame.DataFrame'>
    info = detect.pandas().xyxy[0]

    # .loc y .iloc . La diferencia entre ellos es que .loc acepta etiquetas y .iloc – índices. También cuando usamos los accesores primero especificamos filas y luego columnas.

    # print(info.shape[0]) #numero de filas

    for i in range(info.shape[0]):
        # print(type(info.iloc[[i],[6]]))
        str = info.iat[i, 5]

        x0 = info.iat[i, 0]
        y0 = info.iat[i, 1]
        x1 = info.iat[i, 2]
        y1 = info.iat[i, 3]
        confidence = info.iat[i, 4]
        clase = info.iat[i, 5]
        name = info.iat[i, 6]

        # print(x0)
        # print(x1)
        # print(y0)
        # print(y1)
        # print(confidence)
        # print(clase)
        # print(name)

        # print(type(s))
        # print(s)

        # if name =='train':
        #    print("QUE LLEGA EL TREN¡¡¡¡¡")

        if clase == 0:  # Persona
            x = x1.astype(int)
            y = y1.astype(int)
            punto = [x - 10, y - 10]
            if OSD > 4:
                # correción fernando 28/junio
                cv2.circle(frame, tuple(punto), 1, (255, 255, 0), 3)
            if OSD > 5:
                cv2.imshow(nombre_ventana, np.squeeze(detect.render()))

        if clase == 6:  # Tren
            print("QUE LLEGA EL TREN¡¡¡¡¡")

    # area_pts = np.array([[0, 195], [350, 0], [384, 0], [252, 480], [0, 480]])
    # area_pts = np.array([[0, 1], [10, 0], [14, 0], [12, 10], [0, 40]])

    if utiles.punto_en_poligono(punto, area_pts):
        print("PERSONAS EN LA VIA¡¡¡")
        #playsound(AUDIO_ARCHIVO, False)

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    imagen_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = fgbg.apply(imagen_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, Kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 200:  # 500
            x, y, w, h = cv2.boundingRect(cnt)
            if OSD == 3:
                # dibuja los contornos que se encuentran
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado = 'Alerta Moviemiento'
            color = (0, 0, 255)

    if OSD > 2:
        cv2.drawContours(frame, [area_pts], -1, color,
                         2)  # dibuja area de las vias

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

    if OSD > 1:
        cv2.putText(frame, texto_estado, (340, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "FPS: {:.2f}".format(fpsmax), (340, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.putText(frame, "FPS?: {:.2f}".format(fps1), (10, 120),
        #           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.putText(frame, "Total fotogramas: {:.2f}".format(total_frames), (10, 160),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "N fotos: {:.2f}".format(numero_fotograma), (340, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # if numero_fotograma > 1850:
        #    if not cv2.imwrite('C:\\capturas\\' + str(numero_img) + '.jpg', frame):
        #        raise Exception("No se pudo escribir la imagen")
        #    numero_img += 1

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

    if k == ord('o'):  # OSD On Screen Display
        print("------------OSD = ", OSD)
        OSD += 1
        if OSD > 6:
            OSD = 0

    if k == ord('c'):  # captura de pantalla
        print('C:\\capturas\\' + str(numero_img) + '.jpg')
        # if not cv2.imwrite(os.path.join(os.path.expanduser('~'),'Desktop','tropical_image_sig5.bmp') + str(numero_img) + '.jpg', frame):

        if not cv2.imwrite('C:\\capturas\\' + str(numero_img) + '.jpg', frame):
            raise Exception("No se pudo escribir la imagen")
        numero_img += 1

    # Cerrar ventana desde X
    elif cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_AUTOSIZE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
