import sys
import numpy as np
import cv2
import os

max_boyut = 100
yukseklik=30
genislik=20

im = cv2.imread('training_chars.png')
img = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

imgThreshCopy = thresh.copy()

cv2.imshow("EsikliGoruntu", imgThreshCopy)
    #################      Now finding Contours         ###################

_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,yukseklik * genislik))

responses = []

keys =     [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
            ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
            ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
            ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

for cnt in contours:
        if cv2.contourArea(cnt)>max_boyut:
            [x,y,w,h] = cv2.boundingRect(cnt)

            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roiresized = cv2.resize(roi,(genislik,yukseklik))

            cv2.imshow('Test_Goruntu',im)
            cv2.imshow('RegionofInterest', roi)
            cv2.imshow('ROIResized', roiresized)

            key = cv2.waitKey(0)

            if key == 27:
                    sys.exit()
            elif key in keys:
                    responses.append(int(chr(key)))

                    sample = roiresized.reshape((1,yukseklik * genislik))
                    samples = np.append(samples,sample,0)

floatresponses = np.array(responses,np.float32)

nparesponses = floatresponses.reshape((floatresponses.size,1))

print "training complete"

np.savetxt('gorseldata.txt',samples)
np.savetxt('siniflandirma.txt',nparesponses)

cv2.destroyAllWindows()
