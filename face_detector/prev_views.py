from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request
from cv2 import cv2
import os
import pickle

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

TRAINNERPATH = "{base_path}/trainner.yml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
LABELSPATH = "{base_path}/labels.pickle".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read(TRAINNERPATH)

# read labels from .pickle file
labels = {}
with open(LABELSPATH, "rb") as f:
    labels = pickle.load(f)
    # here we want labelName based on id
    # but our labels are stored like {name: id}
    # so we need to reverse that like {id: name}
    labels = {i: l for l, i in labels.items()}


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

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
            faces = detector.detectMultiScale(
                image,
                scaleFactor=1.2,
                minNeighbors=5,

            )
            #             minSize=(30, 30),
            # flags=cv2.CASCADE_SCALE_IMAGE
            # construct a list of bounding boxes from the detection
            # faces = [(int(x), int(y), int(x + w), int(y + h))
            #          for (x, y, w, h) in faces]
            name = ''
            success = True
            # confidence = 0.0
            (width, height) = (130, 130)
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]

                face_resize_roi = cv2.resize(face_roi, (width, height))
                cv2.imwrite('urlimage.jpg', face_resize_roi)
                id_, confidence = recognizer.predict(face_resize_roi)
                # If confidence is less them 100 ==> "0" : perfect match
                # if (confidence < 500):
                # confidence = int(100*(1-(confidence)/300))
                if (confidence < 500):
                    name = labels[id_]
                    data.update({"success": success, "name": name,
                                 "confidence": confidence})
                    return JsonResponse(data)
                    # confidence = "  {0}%".format(round(100 - confidence))
                else:
                    name = "unknown"
                    success = False
                    data.update({"success": success, "name": name,
                                 "confidence": confidence})
                    return JsonResponse(data)
                # confidence = "  {0}%".format(round(100 - confidence))
                #     # print(id_)
                #     # here we are getting only ids but we need labels to these ids
                #     # so we read them from pickle file
                # name = labels[id_]
        #     # read labels based on id
        #     # print(labels[id_])
            # roiarray.append(roi)

        # update the data dictionary with the faces detected
        # data.update({"num_faces": len(faces), "faces": faces,
        #              "success": True, 'name': name, 'roi': roiarray})
        # data.update({"success": success, "name": name,
        #              "confidence": confidence})

    # return a JSON response
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
