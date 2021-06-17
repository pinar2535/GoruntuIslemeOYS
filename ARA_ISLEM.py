import cv2
import numpy as np
from matplotlib import pyplot as plt

def araislem():
    try:
        new_image = cv2.imread('1.png',1)
    except:
        print("Goruntu dosyadan okunamadi")
        return

    ret, thresh = cv2.threshold(new_image, 60, 255, cv2.THRESH_BINARY)


    gray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)


    cv2.imshow('cer1',new_image)
    cv2.imshow('cer2',gray)
    cv2.imshow('cer3',blur)
    cv2.imshow('cer4', thresh)


    k = cv2.waitKey(0) & 0XFF
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()

    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('yasolii.png', thresh)
        cv2.destroyAllWindows()

    return
araislem()