import os
import numpy as np
from cv2 import cv2
import pickle
from PIL import Image

# os.walk for image finding

face_cascade = cv2.CascadeClassifier(
    "face_detector/cascades/haarcascade_frontalface_default.xml")

# train opencv recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
print('image_dir', image_dir)

current_id = 0  # id for image ROI
label_ids = {}
x_train = []
y_labels = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith('jfif') or file.endswith("jpeg"):
            # if not file.endswith("."):
            path = os.path.join(root, file)
            print(path)
            # getting labels from directories
            label = os.path.basename(os.path.dirname(
                path)).replace(" ", "-").lower()
            print(label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

                id_ = label_ids[label]  # id for each image ROI
                # print(label_ids)
#             # every image has pixel value
#             # train image into numpy array
                pil_image = Image.open(path).convert(
                    "L")  # convert to grayscale

# #             # Resize images for training
                # size = (330, 330)  # size of new image
                # finalImage = pil_image.resize(size, Image.ANTIALIAS)

# #             # convert to numpy array
                image_array = np.array(pil_image, "uint8")
                # print(image_array)

#             # finding ROI in numpy array i.e our training data
                # image = cv2.imread(path)
                # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    image_array,
                    scaleFactor=1.2,
                    minNeighbors=5,


                )
                #   minSize=(30, 30),
                #     flags=cv2.CASCADE_SCALE_IMAGE
                # , scaleFactor=1.5, minNeighbors=5, minSize=(20, 20)
                (width, height) = (130, 130)
                for (x, y, w, h) in faces:

                    face_roi = image_array[y:y+h, x:x+w]
                    face_resize_roi = cv2.resize(face_roi, (width, height))
                    # print('roi', roi)
                    # cv2.imwrite(
                    #     f'trainedimg/{i}.jpg', face_resize_roi)
                    x_train.append(face_resize_roi)
                    y_labels.append(id_)


# Using pickle to save label ids to be used in other modules
with open("face_detector/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
# train opencv recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_detector/trainner.yml")
