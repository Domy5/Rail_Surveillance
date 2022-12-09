import cv2
import numpy as np
import time
import csv

from os import remove
from os import path

fileCSV = 'test_blackgroundSubtractor/test_BS.csv'

if path.exists(fileCSV):
    remove(fileCSV)
    
with open(fileCSV, "a+", newline ='') as csvfile:
        
    wr = csv.writer(csvfile, dialect='excel', delimiter=';')
    wr.writerow(['frameCount', 'mogCount', 'mog2MCount', 'gmgCount', 'knnCount', 'cntCount' ,'frameCount', 'MOGtime', 'MOG2time', 'GMGtime', 'KNNtime', 'CNTtime'])
      

# Video Capture
capture = cv2.VideoCapture("./test_video/a1-003 1 minuto 1 via.mkv")

# Subtractors improved parameter
#mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
#mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
#gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
#knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
#cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# Subtractors default parameter
mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2Subtractor = cv2.createBackgroundSubtractorMOG2()
gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
knnSubtractor = cv2.createBackgroundSubtractorKNN()
cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT()

# Keeps track of what frame we're on
frameCount = 0

# Determine how many pixels do you want to detect to be considered "movement"
movementCount = 1000
movementText = "Movement"
textColor = (255, 255, 255)


while(1):
    # Return Value and the current frame
    ret, frame = capture.read()

    #  Check if a current frame actually exist
    if not ret:
        break

    frameCount += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Sería mejor aplicar apertura morfológica al resultado para eliminar los ruidos. // https://docs.opencv.org/4.x/d8/d38/tutorial_bgsegm_bg_subtraction.html

    # Get the foreground masks using all of the subtractors
    
    inicio = time.time()
    
    mogMask = mogSubtractor.apply(frame)
    
    fin = time.time()
    MOGtime = fin-inicio
    
    inicio = time.time()

    mog2Mmask = mog2Subtractor.apply(frame)
    mog2Mmask = cv2.morphologyEx(mog2Mmask, cv2.MORPH_OPEN, kernel)
    
    fin = time.time()
    MOG2time = fin-inicio
    
    inicio = time.time()
    
    gmgMask = gmgSubtractor.apply(frame)
    gmgMask = cv2.morphologyEx(gmgMask, cv2.MORPH_OPEN, kernel)
        
    fin = time.time()
    GMGtime = fin-inicio
    
    inicio = time.time()
    
    knnMask = knnSubtractor.apply(frame)
    
    fin = time.time()
    KNNtime = fin-inicio
    
    inicio = time.time()
    
    cntMask = cntSubtractor.apply(frame)
    
    fin = time.time()
    CNTtime = fin-inicio


    # Count all the non zero pixels within the masks
    mogCount = np.count_nonzero(mogMask)
    mog2MCount = np.count_nonzero(mog2Mmask)
    gmgCount = np.count_nonzero(gmgMask)
    knnCount = np.count_nonzero(knnMask)
    cntCount = np.count_nonzero(cntMask)

    print('mog Frame: {}, Pixel Count: {}, Time {}'.format(frameCount, mogCount, MOGtime))
    print('mog2M Frame: {}, Pixel Count: {}, Time {}'.format(frameCount, mog2MCount, MOG2time))
    print('gmg Frame: {}, Pixel Count: {}, Time {}'.format(frameCount, gmgCount, GMGtime))
    print('knn Frame: {}, Pixel Count: {}, Time {}'.format(frameCount, knnCount, KNNtime))
    print('cnt Frame: {}, Pixel Count: {}, Time {}'.format(frameCount, cntCount, CNTtime))
    
    #MogCSV = ['MOG',frameCount, mogCount, MOGtime]
    #Mog2CSV = ['MOG2',frameCount, mog2MCount, MOG2time]
    #GmgCSV = ['GMG',frameCount, gmgCount, GMGtime]
    #KnnCSV = ['KNN',frameCount, knnCount, KNNtime]
    #CntCSV = ['CNT',frameCount, cntCount, CNTtime]
    
    CSV = [frameCount, mogCount, mog2MCount, gmgCount, knnCount, cntCount ,frameCount, MOGtime, MOG2time, GMGtime, KNNtime, CNTtime]
    
    with open(fileCSV, "a+", newline ='') as csvfile:
        
        wr = csv.writer(csvfile, dialect='excel', delimiter=';')
        wr.writerow(CSV)
        #wr.writerow(MogCSV)
        #wr.writerow(Mog2CSV)
        #wr.writerow(GmgCSV)
        #wr.writerow(KnnCSV)
        #wr.writerow(CntCSV)

    titleTextPosition = (5, 40)
    titleTextSize = .7
    cv2.putText(mogMask, 'MOG, Frame: {}'.format(frameCount), titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(mog2Mmask, 'MOG2, Frame: {}'.format(frameCount), titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(gmgMask, 'GMG, Frame: {}'.format(frameCount), titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(knnMask, 'KNN, Frame: {}'.format(frameCount), titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(cntMask, 'CNT, Frame: {}'.format(frameCount), titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)

    stealingTextPosition = (0, 80)
    if (frameCount > 1):
        if (mogCount > movementCount):
            print('movementT MOG')
            cv2.putText(mogMask, 'Movement', stealingTextPosition,
                        cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
        if (mog2MCount > movementCount):
            print('movement MOG2')
            cv2.putText(mog2Mmask, 'Movement', stealingTextPosition,
                        cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
        if (gmgCount > movementCount):
            print('movement GMG')
            cv2.putText(gmgMask, 'Movement', stealingTextPosition,
                        cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
        if (knnCount > movementCount):
            print('movement KNN')
            cv2.putText(knnMask, 'Movement', stealingTextPosition,
                        cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
        if (cntCount > movementCount):
            print('movement CNT')
            cv2.putText(cntMask, 'Movement', stealingTextPosition,
                        cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)

    cv2.imshow('Original', frame)
    cv2.imshow('MOG', mogMask)
    cv2.imshow('MOG2', mog2Mmask)
    cv2.imshow('GMG', gmgMask)
    cv2.imshow('KNN', knnMask)
    cv2.imshow('CNT', cntMask)

    cv2.moveWindow('Original', 0, 0)
    cv2.moveWindow('MOG', 640, 0)
    cv2.moveWindow('KNN', 1280, 0)
    cv2.moveWindow('GMG', 0, 500)
    cv2.moveWindow('MOG2', 640, 500)
    cv2.moveWindow('CNT', 1280, 500)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
            break

    # Cerrar ventana desde X
    elif cv2.getWindowProperty('Original', cv2.WND_PROP_AUTOSIZE) < 1:
        break
    # parar si frame = x
   # if frameCount == 750:  
   #     cv2.waitKey(-1)

capture.release()
cv2.destroyAllWindows()
