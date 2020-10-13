import os
import numpy as np
from cv2 import cv2
import pickle
from PIL import Image
import os


face_cascade = cv2.CascadeClassifier(
    "face_detector/cascades/haarcascade_frontalface_alt2.xml")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

labels = []

for root, dirs, files in os.walk(image_dir):
    for dir in dirs:
        path = os.path.join(root, dir)
        dirname = os.path.basename(
            path).replace(" ", "-").lower()
        labels.append(dirname)
print(labels)
# loop through labels for training images
for label in labels:
    # create the train image folder for the user
    if not os.path.exists(os.path.join('images', 'train')):
        os.mkdir(str(image_dir) + "/" + 'train')
    train_path = os.path.join(image_dir, "train")

    if not os.path.exists(os.path.join(train_path, label)):
        os.mkdir(str(train_path) + "/" + label)
    train_path = os.path.join(train_path, label)

    path = os.path.join(image_dir, label)
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith('jfif') or file.endswith("jpeg"):
                # if not file.endswith("."):
                path = os.path.join(root, file)
                print(path)

                grayImage = Image.open(path).convert('L')
                # grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # convert to numpy array
                npGrayImage = np.array(grayImage, "uint8")
                faces = face_cascade.detectMultiScale(
                    npGrayImage, scaleFactor=1.3, minNeighbors=5)
                # faces = face_cascade.detectMultiScale(gray)
                for (x, y, w, h) in faces:
                    # print(x, y, w, h)
                    # region of interest [ycood_start, ycoord_end]
                    roi = npGrayImage[y:y+h, x:x+w]
                    count += 1
                    img_item = os.path.join(train_path, str(count) + ".jpg")
                    cv2.imwrite(img_item, roi)
                # img_item = os.path.join(test_path, str(count) + ".jpg")
