import cv2
import numpy as np
import os
import sys
import operator

max_boyut = 100
yukseklik=30             # yeniden boyutlandirma (vektorel izdusum) icin gerekli parametreler
genislik=20

def main():
    contours = []
    #######   training part    ###############
    try:
        samples = np.loadtxt('ZZimagedata.txt',np.float32) #sayilarin ve harflerin datalarini yukleme
    except:
        print("HATA, ZZimagedata.txt acilamiyor, programdan cikiliyor")
        os.system("pause")
        return
    try:
        responses = np.loadtxt('ZZclass.txt',np.float32) # siniflandirma datalarini yukleme
    except:
        print("HATA, ZZclass.txt acilamiyor, programdan cikiliyor")
        os.system("pause")
        return

    responses = responses.reshape((responses.size,1)) # train asamasina uyumlu hale sokmak icin yeniden boyutlandirma islemi

    kNearest = cv2.ml.KNearest_create()                   # KNN algoritmasini cagirma islemi

    kNearest.train(samples, cv2.ml.ROW_SAMPLE, responses) # KNN algoritmasi Train asamasi

############################# testing part  #########################

    im = cv2.imread('yasolii.png') # gorseli program tarafindan okuma
    if im is None:
        print("HATA, gorsel dosyadan okunamadi")
        os.system("pause")
        return

    out = np.zeros(im.shape,np.uint8) #daha sonrasi icin gerekli olan bir cekirdek matris olusuturuluyor

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #griye cevirme islemi
    #cv2.imshow("gri", gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #blurlama islemi (gurultu azaltmak icin)
    #cv2.imshow("blur",blurred)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #esikleme islemi
                                #(islem yapilacak goruntu,max. deger,adaptif thresh yontemi, thresh tipi, komsu piksellere gore threshold yapmak icin parametre)
    #ret , thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    #cv2.imshow("thresh",thresh)
    imgThreshCopy = thresh.copy() # kopya threshold gorseliyle devam ediliyor

    _,contours,hierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #kontur bulma islemi

    sorted_ctrs = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0]) #### konturleri soldan saga siralama islemi
                                                                            #### 0' i 1 yaparsak sagdan sola siraliyor
    #contours.sort(key=operator.attrgetter("x"))

    strFinalString = ''           # bos bir parametre actik, ilerde  doldurulacak
    #h = 0
    for cnt in sorted_ctrs:     # her kontur icin tarama yapilacak dongu
            if cv2.contourArea(cnt)>max_boyut:    # tanimladigimiz max boyuttan kucuk alanlar olmasi icin
                [x,y,w,h] = cv2.boundingRect(cnt)  #dikdortgen cizmek icin x,y orijin parametreleri , w=genislik,h=yukseklik parametreleri
            if  h>28  :                              #28 degerinden yuksek konturler bulup dikdortgen cizmesini saglamak icin
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)     # kirmizi bir dikdortgen ciziyor, (orijinal resmin ustune,baslangic pikseli,bitis pikseli,yesil renkte,kalinlik)

                roi = thresh[y:y+h,x:x+w] #thresholdlu goruntudeki karakterleri disariya almak icin
                roismall = cv2.resize(roi,(genislik,yukseklik)) #goruntuyu yeniden boyutlandiriyor, karsilastirma yapabilmek icin ayni boyutta olamalari gerek
                roismall = roismall.reshape((1,genislik*yukseklik)) #boyutlanmis goruntuyu numpy dizisine ceviriyor
                roismall = np.float32(roismall) # int degerdeki numpy arrayleri float degerine ceviriyor

                retval, results, neigh_resp, dists = kNearest.findNearest(roismall, k = 1) # KNN fonksiyonu cagiriliyor, en yakin komsular bulunuyor

                 #string = str((int((results[0][0]))))


                strCurrentChar = str(chr(int(results[0][0])))  # sonuclardan harf ve sayilari alabilmek icin

                strFinalString = strFinalString + strCurrentChar # bos parametreyi harf ve sayi bilgileriyle doldurduk

                cv2.putText(out, strCurrentChar, (x, y + h), 0, 1, (0, 255, 0),1) # daha onceden olusan cekirdek matrise bu yazi degerlerini atiyor
                #END IF , END IF , END FOR                                         ## putText(cekirdek matris,icine gelcek yazi,baslangic ve bitisi,fontFace,font Olcek,yazi rengi,Kalinlik)

    print ("Tespit edilen Plaka :\t" + strFinalString + "\t") #elde edilen karakterleri (sayi ve harf) gosterme islemi
    cv2.imshow('im',im) # orijinal resmi gosterme ( cizilen dikdortgenle)
    cv2.imshow('out',out) # puttext komutuyla icini doldurdugumuz pencereyi gosterme

    cv2.waitKey(0) # bekletme
    cv2.destroyAllWindows()
    return
############################# CALISTIRMA ##############################
#if name == "main":
main()
