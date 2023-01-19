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

import os
import argparse
import cv2
import sys
import datetime
import torch
import time
import csv

from playsound import playsound
import numpy as np
import pandas as pd

from os import remove
from os import path

import tools.utils as utils
import db.controler as controler

###############################

#  Variable

loop_flag = 0
pos = 0
frame_number = 0
fpsmax = 0
counter = 0
img_number = 0
windows_name = 'Rail_Surveillance 0.75'
points_list = []
point = [0, 0]
frame_counter = 0
contour_quantity = 0

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

flag_1 = True
flag_2 = True
flag_3 = True
flag_4 = True
flag_5 = True
flag_6 = True
flag_7 = True

fileCSV = 'system_evaluation/eval.csv'

if path.exists(fileCSV):
    remove(fileCSV)
    
with open(fileCSV, "a+", newline ='') as csvfile:
        
    wr = csv.writer(csvfile, dialect='excel', delimiter=';')
    wr.writerow(['frame_number', 'timer_total', 'fps', 'fps1',  'contour', 'train', 'person_on_via'])

utils.clear_screen()

###############################

#  Parameters

parser = argparse.ArgumentParser(description=' Detention of people and/or objects on the railway platform in real time ')

parser.add_argument('-v', '--version', action="store_true",default=False, help='Program version')
parser.add_argument('-info', '--information', action="store_true",default=False, help='Information about the versions of the packages used')
parser.add_argument('-m', '--mask', action="store_true",default=True, help='Show the mask')
parser.add_argument('-d', '--detection', action="store_true",default=False, help='Shows object detections')
parser.add_argument('-s', '--slicer', action="store_true", default=False,help='Show scrollbar (consumes a lot of resources)')
parser.add_argument('-mm', '--mouse', action="store_true", default=False,help='Displays the coordinates of the clicks on the console')
parser.add_argument('-c', '--process_image',
                    type=str,
                    choices=['gpu', 'cpu'],
                    default='gpu',  # default='cpu',
                    required=False,
                    help='parameter GPU or CPU')
parser.add_argument('-i', '--input',
                    type=str,
                    default='1', # Change of video camera and its ROI
                    required=False,
                    help='Administrative code of the station')

args = parser.parse_args()

if args.version:
    print(windows_name)
    sys.exit("See you soon!")

if args.information:
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

    args.process_image = 'cpu'

    print('No GPU enabled, CPU will be used for image processing')
    print(' ')

if args.input:

    path = controler.get_video_path(args.input)
    polygon = controler.get_video_ROI(args.input)

    cap = cv2.VideoCapture(f'{path}')
    area_points = np.array(polygon)
    
if args.process_image == 'gpu':
    gpu_frame = cv2.cuda_GpuMat()  # to use GPU
    
###############################

#  Object Classifier (Model)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', force_reload=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True, force_reload=True)

# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
# trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
#####    enabled_precisions = {torch.half},
#####    workspace_size = 1 << 22
# )

# print(model.eval())

# (optional list) filter by class, i.e. = [0, 15, 16] for COCO
# 0 = person, 1 = bike, 6 = train, 36 = skateboard, 26 = handbag, 16 = dog, 24  backpack, 13    bench, 28  suitcase
model.classes = [0, 6]
model.conf = 0.3  # NMS confidence threshold   
model.iou = 0.45  # NMS IoU threshold (0-1)  https://kikaben.com/object-detection-non-maximum-suppression/
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

###############################

#  Menu

cv2.namedWindow(windows_name)
cv2.moveWindow(windows_name, 10, 50)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cap.get(cv2.CAP_PROP_FPS))

seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("#######################")
print("## Execution in : {} ##".format(args.process_image))
print("#######################")
print("Video name: {} ".format(args.input))
print("Duration in seconds :", seconds)
print("Video time          :", video_time)

print("HEIGHT: {}".format(height))
print("WIDTH: {}".format(width))

print("Nº Frame: {}".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("")
print('Options:')
print('------------------------------')
print('- Esc :-> Close the execution of the video')
print('- p   :-> Stop the video')
print('- c   :-> Captures a frame from the video and saves it to \\screenshots\\number_img.jpg')
print('- s   :-> Activate alarm sound')
print('')
print('------------------------------')
print('')
print('OSD:')
print('------------------------------')
print('- 1 :-> Infor, Alarm, FPS, Frame number...')
print('- 2 :-> ROI')
print('- 3 :-> Outlines inside ROI by background subtraction')
print('- 4 :-> Right foot point in people')
print('- 5 :-> Contours in the scene (People, trains, bags, cars)')
print('- 6 :-> Enable better performance')
print('------------------------------')
print('')

# utiles.clear_screen()

if args.slicer:
    cv2.createTrackbar('time', windows_name, 0, frames, utils.nothing)
        
if args.mouse:
    cv2.setMouseCallback(windows_name, utils.draw_dots)

# Subtractors

# mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
# mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, False)
# gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
# knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
# cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG( )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 200,nmixtures = 5, backgroundRatio = 0.7,noiseSigma = 0 )
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()

mog2Subtractor = cv2.createBackgroundSubtractorMOG2()

Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

###############################

#  Detection

while True:
    
    start = time.time()
    
    frame_number += 1
    
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
    
    contour = False
    train = False
    person_on_via = False

    #  Enable better performance
    if not flag_6:
         frame_counter += 1
         if frame_counter % 2 !=0:
             print("Enable better performance {}....{}".format(flag_6, frame_counter))
             continue

    if args.process_image == 'gpu':
        gpu_frame.upload(frame)                                     # to use GPU
        gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)     # to use GPU
        gray = gray.download()                                      # to use GPU
    elif args.process_image == 'cpu':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("Image processor error!")

    # detect = model(frame, size = 640)
    detect = model(frame)
    
    #detect.print()
        #Speed: 11.0ms pre-process, 59.3ms inference, 3.0ms NMS per image at shape (1, 3, 512, 640)
        #image 1/1: 480x640 5 persons

    # info is an object of type <class 'pandas.core.frame.DataFrame'>
    info = detect.pandas().xyxy[0]

    # .loc y .iloc . The difference between them is that .loc accepts labels and .iloc – indices. Also when we use the accessors we first specify rows and then columns.
    # print(info.shape[0]) # number of rows

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
            
            # We paint a dot in the lower right corner of each detected person
            x = x1.astype(int)
            y = y1.astype(int)
            point = [x - 10, y - 10]
            if flag_4:
                cv2.circle(frame, tuple(point), 1, blue_color_cyan, 3)

        if clase == 6:  # Train
            train = True
            
        if flag_5:
            cv2.imshow(windows_name, np.squeeze(detect.render()))

    if utils.point_in_polygon(point, area_points):
        person_on_via = True

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = mog2Subtractor.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, Kernel)  # It would be better to apply morphological aperture to the result to remove the noises.
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in cnts:
        if cv2.contourArea(cnt) > 250:  # 500
            x, y, w, h = cv2.boundingRect(cnt)
            if flag_3:
                # Draw the contours that meet
                cv2.rectangle(frame, (x, y), (x + w, y + h), yellow_color, 2)
                contour = True
    
###############################

#  Info FPS
    
    end = time.time()

    fps = 1 / (end - start)
    fps1 = cap.get(cv2.CAP_PROP_FPS)

    # Get number of video frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # So that time to see the fps
    if counter < 5:
        counter = counter + 1
    else:
        fpsmax = fps
        counter = 0

###############################

#  Logic .. if there is a train .. if there are people on the track, if an object has fallen

    status_text = 'Motionless'
    color = green_color

    if contour :
        contour_quantity +=1
        
        if contour_quantity == 5:
            status_text = 'Caution Movement'
            color = yellow_color
            contour_quantity = 0
            utils.play_track('assets/sound', 'alarm_2.wav', flag_7)
            #print("Contorno -------------------------------------------- {}".format(frame_number))
        
    if person_on_via :
        status_text = 'ALERT Movement'
        color = red_color
        print("PEOPLE ON THE TRAIN TRACK¡¡")
        utils.play_track('assets/sound', 'alarm_1.wav', flag_7)
        #print("Persona -------------------------------------------- {}".format(frame_number))
        
    if train and person_on_via :
        status_text = 'ALERT Movement'
        color = red_color
        print("ARRIVAL OF THE TRAIN WITH PERSON ON THE TRAIN TRACK¡¡")
            
    if train : # Remove contours if the train is present
        flag_3 = False
        #print("Tren -------------------------------------------- {}".format(frame_number))
    else:
        flag_3 = True

###############################

#  Options

    if flag_1:
    
       cv2.putText(frame, status_text, (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       
       if flag_7:
            cv2.putText(frame, "Sound", (550,18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2)
       else:
            cv2.putText(frame, "Sound", (550,18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
       
       cv2.putText(frame, 'FPS: {:.2f}'.format(fpsmax), (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       cv2.putText(frame, 'N frame: {:.2f}'.format(frame_number), (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)      
    
    if flag_2:
        cv2.drawContours(frame, [area_points], -1, color, 2)  # draw ROI
        cv2.drawContours(fgmask, [area_points], -1, gray_color, 2) # Draw ROI in Background Subtraction in ROI
        
    if args.mask:
        cv2.imshow('Background Subtraction in ROI', fgmask)
        # cv2.moveWindow('fgmask', 10, 75 + height);

    if args.detection:
        cv2.imshow(windows_name, np.squeeze(detect.render()))
        print(detect.render())
        
    # Show info FPS
    
    cv2.imshow(windows_name, frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:                 # Close window from Esc
        break

    if k == ord('p'):           # Stop if you press p
        cv2.waitKey(-1)

    if k == ord('c'):           # Screenshots
       if os.name == "ce" or os.name == "nt" or os.name == "dos":
           try:
               os.stat('C:\\screenshots\\')
           except:
               os.mkdir('C:\\screenshots\\')            
           print('C:\\screenshots\\' + str(img_number) + '.jpg')
           if not cv2.imwrite('C:\\screenshots\\' + str(img_number) + '.jpg', frame):
               raise Exception("Could not write image")
           img_number += 1        
       elif os.name == "posix":
           try:
               os.stat('/screenshots/')
           except:
               os.mkdir('/screenshots/')             
           print('/screenshots/' + str(img_number) + '.jpg')
           if not cv2.imwrite('/screenshots/' + str(img_number) + '.jpg', frame):
               raise Exception("Could not write image")
           img_number += 1
       else:
           print("unknown OS")
        
    #  OSD On Screen Display

    if k == ord('1'):                       # Infor, Alarm, FPS, Frame number...
        flag_1 = False if flag_1 else True
    if k == ord('2'):                       # ROI
        flag_2 = False if flag_2 else True   
    if k == ord('3'):                       # Outlines inside ROI by background subtraction
        flag_3 = False if flag_3 else True      
    if k == ord('4'):                       # Right foot point in people  
        flag_4 =  False if flag_4 else True   
    if k == ord('5'):                       # Contours in the scene (People, trains, bags, cars)
        flag_5 = False if flag_5 else True    
    if k == ord('6'):                       # Enable better performance'
        flag_6 = False if flag_6 else True    
    if k == ord('s'):                       # Activate alarm sound
        flag_7 = False if flag_7 else True
    elif cv2.getWindowProperty(windows_name, cv2.WND_PROP_AUTOSIZE) < 1: # Close window from X
        break
    
###############################

    end2 = time.time()

    timer_total =(end2 - start)

#  File CSV eval
    
    #CSV = [frame_number, mogCount, mog2MCount, gmgCount, knnCount, cntCount ,frameCount, MOGtime, MOG2time, GMGtime, KNNtime, CNTtime]
    
    str_timer_total = str(timer_total).replace(".",",")
    str_fps = str(fps).replace(".",",")
    str_fps1 = str(fps1).replace(".",",")
    
    CSV = [frame_number,str_timer_total ,str_fps, str_fps1,  contour, train, person_on_via ] #, mogCount, mog2MCount, gmgCount, knnCount, cntCount ,frameCount, MOGtime, MOG2time, GMGtime, KNNtime, CNTtime]
    
    with open(fileCSV, "a+", newline ='') as csvfile:
        
        wr = csv.writer(csvfile, dialect='excel', delimiter=';')
        wr.writerow(CSV)
        
###############################

#  Exit

cap.release()
cv2.destroyAllWindows()

##############################