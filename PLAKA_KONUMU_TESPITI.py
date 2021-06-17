import cv2
import numpy as np
from matplotlib import pyplot as plt

img2 = cv2.imread('3.jpg',1) #goruntuyu renkli sekilde programa okutur.(goruntu dosyasinin adi,renkli okuanacagi=1)

#img = cv2.imread('1.jpg',0) #goruntuyu gri tonlamali sekilde programa okutur ( alternatif)

gri = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)### renkli okutulup bu islemle gri tonlamaya donusturur.

gurultuazalt = cv2.bilateralFilter(gri, 9, 75, 75)

histogram_e = cv2.equalizeHist(gurultuazalt)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

morfolojikresim = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel, iterations=15)

gcikarilmisresim = cv2.subtract(histogram_e, morfolojikresim)

ret, goruntuesikle = cv2.threshold(gcikarilmisresim, 0, 255, cv2.THRESH_OTSU)
#ret,thresh1 = cv2.threshold(gcikarilmisresim,100,200,cv2.THRESH_BINARY_INV) gereksiz

canny_goruntu = cv2.Canny(goruntuesikle, 250, 255)

canny_goruntu = cv2.convertScaleAbs(canny_goruntu)

cekirdek = np.ones((3, 3), np.uint8)

gen_goruntu = cv2.dilate(canny_goruntu, cekirdek, iterations=1)

new, contours, hierarchy = cv2.findContours(gen_goruntu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   ##TUM KONTURLER BULUNDU

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]                     ##KONTURLER SIRALANDI

screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)              ##BULUNAN KONTURLERDE KAPALI BIR DIKDORTGEN BULMA ALGORITMASI

    if len(approx) == 4:
        screenCnt = approx
        break

final = cv2.drawContours(img2, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gri.shape ,np.uint8)                           #SIYAH BIR ARKA PLAN YARATTIK

# get first masked value (foreground)
#fg = cv2.bitwise_or(img2, img2, mask=mask)                              #SIYAH ARKA PLAN
# get second masked value (background) mask must be inverted
#mask = cv2.bitwise_not(mask)                                   #MASKE ARKAPLANI BEYAZ OLDU
#background = np.full(img2.shape,255, dtype=np.uint8)            #BEYAZ ARKA PLAN OLUSTURDUK
#bk = cv2.bitwise_or(background, background, mask=mask) #          #BEYAZ ARKA PLAN

# combine foreground+background
#finalZZ = cv2.bitwise_and(fg, bk)                            #SIYAH=0, BEYAZ = 1 OR ISLEMI ARKA PLAN BEYAZ OLUR

yeni_goruntu = cv2.drawContours(mask, [screenCnt], 0,255, -1)               ##-1 olmasi, butun plaka alaninin icini beyaz yapiyor demek
#finalAA = cv2.bitwise_not(yeni_goruntu)                                     ## PLAKA ICI SIYAH, ARKA PLAN BEYAZ OLDU

yeni_goruntu = cv2.bitwise_and(img2, img2, mask=mask)       #bitwise_not arka plan orjinal goruntu kaliyor, kontur ici negatif filtre oluyor

#finalAb =cv2.bitwise_or(img2, img2, mask=finalAA)          ##ARKA PLAN ARABA, PLAKA ICI SIYAH OLDU

#QQQ = cv2.bitwise_not(finalAA)

y, cr, cb = cv2.split(cv2.cvtColor(yeni_goruntu, cv2.COLOR_RGB2YCrCb))
y = cv2.equalizeHist(y)
son_resim = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


cv2.imshow("1", img2)
cv2.imshow("2", gri)
cv2.imshow("3", gurultuazalt)
cv2.imshow("4", histogram_e)
cv2.imshow("5",morfolojikresim )
cv2.imshow("6",gcikarilmisresim )
cv2.imshow("7", goruntuesikle)                      #####Her adimi pencerede gosteriyor.
cv2.imshow("8", canny_goruntu)
cv2.imshow("9", canny_goruntu)
cv2.imshow("10", gen_goruntu)
cv2.imshow("11", final)
cv2.imshow("mask", mask)
cv2.imshow("12", yeni_goruntu)
cv2.imshow("13", son_resim)

#cv2.imshow("zz", finalZZ)
#cv2.imshow("bk", bk)
#cv2.imshow("fg", fg)
#cv2.imshow("back",background)
#cv2.imshow("finalAA", finalAb)
#cv2.imshow("QQQ", QQQ)

k = cv2.waitKey(0) & 0XFF
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()

elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('3saved3.png',yeni_goruntu)
    cv2.destroyAllWindows()
