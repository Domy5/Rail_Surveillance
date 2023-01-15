
import cv2
import numpy as np

def dibujar(event, x, y, etiquetas,parametros):
    if event == cv2.EVENT_MOUSEMOVE:
        a = "{}, {}".format(x,y)
        cv2.putText(i, a, (x+5, y+5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.namedWindow(winname = 'mi')
cv2.setMouseCallback('mi',dibujar)



while True:
    i = np.zeros((500,500,3), np.int8)
    cv2.imshow( 'mi', i)
    
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()