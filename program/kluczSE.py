# zaladowanie bibliotek
from ultralytics import YOLO
import torch
# import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

# przetwarzanie argumentow programu
parser = argparse.ArgumentParser(
                    prog='kluczSE.py',
                    description='Program wykrywajacy klucze plasko-oczkowe')

parser.add_argument('source',help='zrodlo obrazu (0 dla kamerki)')
parser.add_argument('-g', '--gpu', action='store_true', help='obliczenia na GPU (tylko Nvidia)')

args = parser.parse_args()

if args.source == '0':                      # program dla kamerki
    model = YOLO('bestn.pt')                # wykorzystany model to yolov5nu
    capture = cv2.VideoCapture(0)           # przechwycenie obrazu z kamerki
    while capture.isOpened():
        ret,image = capture.read()
        if ret == True:                     # jesli jest dostrpna klatka można przetworzyc
            
            if args.gpu:                                                    # wykorzystanie GPU do obliczen
                results = model(image, device='0')
            else:                                                           # wykorzystanie CPU do obliczen
                results = model(image)
            bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")     # tablica z pozycjami prostokatow
            confs = np.array(results[0].boxes.conf.cpu(), dtype="float")     # tablica z wartosciami pewnosci wykrytych obiektow

            for conf, bbox in zip(confs, bboxes):
                (x, y, x2, y2) = bbox
                cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 225), 2)      # rysowanie prostokata na obrazie
                cv2.putText(image, str(round(conf,2)), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
            cv2.imshow('Obraz z kamery', image)                             # wyswietlenie obrazu
            if cv2.waitKey(25) & 0xFF == ord('x'):                          # zatrzymanie programu po wcisnieciu x
                break
        else:
            break
    capture.release()           # zwolnienie zasobu kamerki

else:                                                   # program dla pliku
    model = YOLO('best.pt')                             # wykorzystany model to yolov5mu
    image = cv2.imread(args.source)                     # odczyt obrazu z pliku
    
    if image.shape[0] > 640 or image.shape[1] > 800:    # skalowanie obrazu jesli jest za duzy
        scale = min(640.0/image.shape[0], 800.0/image.shape[1])
        print(scale)
        size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image1 = image
        image = cv2.resize(image1, size, cv2.INTER_LANCZOS4)
    if args.gpu:                                                            # wykorzystanie GPU do obliczeń
        results = model(image, device='0')
    else:                                                                   # wykorzystanie CPU do obliczen
        results = model(image)

    bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")             # tablica z pozycjami prostokatow
    confs = np.array(results[0].conf.cls.cpu(), dtype="float")             # tablica z wartosciami pewnosci wykrytych obiektow

    for conf, bbox in zip(confs, bboxes):
       (x, y, x2, y2) = bbox
       cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 225), 2)               # rysowanie prostokata na obrazie
       cv2.putText(image, str(round(conf,2)), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    cv2.imshow("wynik", image)                                              # wyswietlenie obrazu

    cv2.waitKey(0)                                                          # oczekiwanie na interakcje

    cv2.destroyAllWindows()     # zamkniecie okien