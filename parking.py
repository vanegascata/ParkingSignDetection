import cv2 
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import numpy as np
import os
from PIL import Image
import pytesseract
from pytesseract import Output
import torch
from tkinter import Tk

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.pt') 

# Distancia Lev
def lev_dist(a, b):

    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )
    return min_dist(0, 0)

# Definición del diccionario
def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False

new_dict = ['Bewohner', 'mit', 'Parkschein', 'Parkausweis','frei','Mo-Fr', 'Mo', 'Fr','Mo - Fr','Sa', 'automat', '8-20h', '8-22h', 'Std', 'Std.']
numbers_list = ['8-20h']
n_words = len(new_dict)

# BORRAR RESULTADOS ANTERIORES
palabras_encontradas = []

for f in os.listdir(r"Results/crops/Sign/"):
    if not f.endswith(".jpg"):
        continue
    os.remove(os.path.join("Results/crops/Sign/", f))

for f in os.listdir(r"Results"):
    if not f.endswith(".jpg"):
        continue
    os.remove(os.path.join("Results", f))

from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
im = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(im)

#Ejecutar la predicción de la imagen
results = model(im)
results.crop(save=True,save_dir='Results')

file_list=os.listdir(r"Results/crops/Sign/")
image_path = os.path.join("Results/crops/Sign/", file_list[0])
# print(image_path)
img = cv2.imread(image_path)
img_original=img.copy()
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


## PRE PROCESSING IMAGE
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, img_bw = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
new_img = img_bw.astype(np.uint8)

# FIND CONTOURS
#im2, cnt, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) - not enough values to unpack (expected 3, got 2)
cnt, hierarchy = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(cnt)))
sorted_cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

# #The follwing lines modify img as well
img_cnt = cv2.drawContours(img, cnt, -1, 255, 5)

x = [5]
y = [5]
w = [5]
h = [5]

# Crop Contours
x,y,w,h = cv2.boundingRect(sorted_cnt[1])
cv2.rectangle(img_cnt,(x,y),(x+w,y+h),(180,255,0),30)
cropped_A = new_img[y:y+h, x:x+w]

x,y,w,h = cv2.boundingRect(sorted_cnt[2])
cv2.rectangle(img_cnt,(x,y),(x+w,y+h),(180,255,0),30)
cropped_B = new_img[y:y+h, x:x+w]

x,y,w,h = cv2.boundingRect(sorted_cnt[3])
cv2.rectangle(img_cnt,(x,y),(x+w,y+h),(180,255,0),30)
cropped_C = new_img[y:y+h, x:x+w]

x,y,w,h = cv2.boundingRect(sorted_cnt[4])
cv2.rectangle(img_cnt,(x,y),(x+w,y+h),(180,255,0),30)
cropped_D = new_img[y:y+h, x:x+w]

x,y,w,h = cv2.boundingRect(sorted_cnt[5])
cv2.rectangle(img_cnt,(x,y),(x+w,y+h),(180,255,0),30)
cropped_E = new_img[y:y+h, x:x+w]

plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(img_cnt, cv2.COLOR_BGR2RGB))

cropped_list = [cropped_A, cropped_B, cropped_C, cropped_D, cropped_D, cropped_E]


# Extracción de palabras
custom_config = r'--oem 3 --psm 11'
for cropped_X in cropped_list:
   image_to_read = cropped_X

   d = pytesseract.image_to_data(image_to_read, config=custom_config, lang='deu', output_type=Output.DICT)
   # Data to Text
   n_boxes = len(d['text'])
   #print('Palabras encontradas:')

   for i in range(n_boxes):
      if int(float(d['conf'][i])) > 0:
            for j in range(n_words):
               if (lev_dist(new_dict[j],d['text'][i]) <= 1):
                  #print(new_dict[j],' - ', d['conf'][i])
                  palabras_encontradas.append(new_dict[j])
                  (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                  image_to_read = cv2.rectangle(image_to_read, (x, y), (x + w, y + h), (0, 255, 0), 2)

   d = pytesseract.image_to_data(image_to_read, config=custom_config, lang='deu', output_type=Output.DICT)
   # Data to Text
   n_boxes = len(d['text'])
   #print('Palabras encontradas:')

   for i in range(n_boxes):
      if int(float(d['conf'][i])) > 0:
            for j in range(n_words):
               if (lev_dist(new_dict[j],d['text'][i]) <= 1):
                  #print(new_dict[j],' - ', d['conf'][i])
                  palabras_encontradas.append(new_dict[j])
                  (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                  image_to_read = cv2.rectangle(image_to_read, (x, y), (x + w, y + h), (0, 255, 0), 2)

pal_enc = set(palabras_encontradas)
print('Palabras encontradas:')
print(pal_enc)


# Logica y Respuesta
import datetime as dt
now = dt.datetime.now()

Bewohner = False
Dia = False
Std = False
Parkschein = False
Respuesta = True

if 'Bewohner' in pal_enc or 'Parkausweis' in pal_enc or 'mit' in pal_enc:
    Respuesta = False
    Bewohner = True

hour = now.hour
weekday = now.weekday()
#weekday = 3
if 'Mo-Fr' in pal_enc or 'Sa' in pal_enc or 'Fr' in pal_enc or 'Mo' in pal_enc:
    if 'Mo-Fr' in pal_enc and weekday < 5:
        if '8-20h' in pal_enc and (hour < 20 and hour>8):
            Dia = True
    elif 'Sa' in pal_enc and weekday == 5:
        Dia = True
        Respuesta = True

if 'Std.' in pal_enc or 'Std' in pal_enc:
    Std = True
    Respuesta = True

if 'Parkschein' in pal_enc:
    Parkschein = True
    Respuesta = True

# print("Bewohner:", Bewohner)
# print("Dia:", Dia)
# print("Std:", Std)
# print("Parkschein:", Parkschein)
# print("Respuesta:", Respuesta)

#RESPUESTA FINAL
if Respuesta:
    print("Sí puedes  estacionar :), pero:")
    if Parkschein:
        print("Debes comprar el tiquete")
    if Std:
        print("Debes usar el parquímetro")
    if Dia:
        print("Solo en las horas indicadas")
else:
    print("No puedes estacionar :(")