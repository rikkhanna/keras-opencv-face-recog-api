# opencv and Keras API for face recognition

##Description
Face-Detection using OpenCV library and Tensorflow with Django python server.

## Running Server:

Go to root directory of the project, and

```bash
$ python manage.py runserver 127.0.0.1:5050
```

## POST request using postman:

- Posting image via URL:
  make POST request 'http://localhost:8000/face_detection/detect/'
  pass body -> form-data -> 'url=https://upload.wikimedia.org/wikipedia/commons/9/90/PM_Modi_2015.jpg'
