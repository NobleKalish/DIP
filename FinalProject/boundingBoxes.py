import numpy as np
import cv2
import os
import csv

src = r'./data2/val'

src_files = os.listdir(src)

fields = ['path', 'x1', 'y1', 'x2', 'y2']

csv_file = 'boundingBoxes_val.csv'

with open(csv_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

for file_name in src_files:
    full_directory_name = os.path.join(src, file_name)
    src_folder = os.listdir(full_directory_name)
    for file in src_folder:
        full_image_name = os.path.join(full_directory_name, file)
        img = cv2.imread(full_image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        with open(csv_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([file, x, y, x+w, y+h])
