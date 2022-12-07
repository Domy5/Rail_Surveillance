import cv2
import numpy as np

# Video Capture
capture = cv2.VideoCapture("./test_video/a1-003 1 minuto 1 via.mkv")

# Subtractors
mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

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
    # Resize the frame
    # resizedFrame = cv2.resize(frame, (0, 0), fx=0.375, fy=0.25)
    resizedFrame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Get the foreground masks using all of the subtractors
    mogMask = mogSubtractor.apply(resizedFrame)
    mog2Mmask = mog2Subtractor.apply(resizedFrame)
    gmgMask = gmgSubtractor.apply(resizedFrame)
    gmgMask = cv2.morphologyEx(
        gmgMask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    knnMask = knnSubtractor.apply(resizedFrame)
    cntMask = cntSubtractor.apply(resizedFrame)

    # Count all the non zero pixels within the masks
    mogCount = np.count_nonzero(mogMask)
    mog2MCount = np.count_nonzero(mog2Mmask)
    gmgCount = np.count_nonzero(gmgMask)
    knnCount = np.count_nonzero(knnMask)
    cntCount = np.count_nonzero(cntMask)

    print('mog Frame: %d, Pixel Count: %d' % (frameCount, mogCount))
    print('mog2M Frame: %d, Pixel Count: %d' % (frameCount, mog2MCount))
    print('gmg Frame: %d, Pixel Count: %d' % (frameCount, gmgCount))
    print('knn Frame: %d, Pixel Count: %d' % (frameCount, knnCount))
    print('cnt Frame: %d, Pixel Count: %d' % (frameCount, cntCount))

    titleTextPosition = (5, 40)
    titleTextSize = .7
    cv2.putText(mogMask, 'MOG', titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(mog2Mmask, 'MOG2', titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(gmgMask, 'GMG', titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(knnMask, 'KNN', titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(cntMask, 'CNT', titleTextPosition,
                cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)

    stealingTextPosition = (100, 40)
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

    cv2.imshow('Original', resizedFrame)
    cv2.imshow('MOG', mogMask)
    cv2.imshow('MOG2', mog2Mmask)
    cv2.imshow('GMG', gmgMask)
    cv2.imshow('KNN', knnMask)
    cv2.imshow('CNT', cntMask)

    cv2.moveWindow('Original', 0, 0)
    cv2.moveWindow('MOG', 640, 0)
    cv2.moveWindow('KNN', 1280, 0)
    cv2.moveWindow('GMG', 0, 480)
    cv2.moveWindow('MOG2', 640, 480)
    cv2.moveWindow('CNT', 1280, 480)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
            break

    # Cerrar ventana desde X
    elif cv2.getWindowProperty('Original', cv2.WND_PROP_AUTOSIZE) < 1:
        break

capture.release()
cv2.destroyAllWindows()