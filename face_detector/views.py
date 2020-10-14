from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from tensorflow.keras.models import load_model
import numpy as np
import urllib.request
from cv2 import cv2
import os
from PIL import Image


# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_alt2.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trained_model_path = os.path.join(BASE_DIR, "face_recognizer_model.h5")
# print(trained_model_path)
IMAGE_SIZE = 120

model = load_model(trained_model_path)


face_cascade = cv2.CascadeClassifier(
    FACE_DETECTOR_PATH)  # pylint: disable=no-member

labels = os.listdir("./Images/train/")


@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False, "name": "Not Recognized"}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
            # cv2.imwrite('urlimage.jpg', image)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
            faces = detector.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=5,

            )

            name = ''
            success = True
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]

                face = cv2.resize(face_roi, (IMAGE_SIZE, IMAGE_SIZE))

                img = Image.fromarray(face, mode='RGB')
                # img = Image.frombytes('RGBA', (128, 128), face, 'raw')

                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                # cv2.imwrite('urlimage.jpg', img_array)
                prediction = model.predict(img_array)
                print(prediction)
                index = np.argmax(prediction)
                print(index)
                name = labels[index]
                data.update({"success": success, "name": name})
                # return JsonResponse(data)
    return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image
