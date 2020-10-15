# import numpy as np
# import time

# import PIL.Image as Image
# import matplotlib.pylab as plt

# import tensorflow as tf
# import tensorflow_hub as hub


# classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
# IMAGE_SHAPE = (224, 224)

# classifier = tf.keras.Sequential([
#     hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
# ])

# # download a single image to try the model on

# rishabh = tf.keras.utils.get_file(
#     'Rishabh.jpg', 'https://pbs.twimg.com/profile_images/1092976378792730624/KGKuRpmL_400x400.jpg')
# rishabh = Image.open(rishabh).resize(IMAGE_SHAPE)
# rishabh = np.array(rishabh)/255.0
# # print(rishabh.shape)

# # add a batch dimension and pass the image to model

# result = classifier.predict(rishabh[np.newaxis])
# # print(result.shape)
# predicted_class = np.argmax(result[0], axis=-1)
# # print(predicted_class)

# labels_path = tf.keras.utils.get_file(
#     'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())

# plt.imshow(rishabh)
# plt.axis('off')
# predicted_class_name = imagenet_labels[predicted_class]
# _ = plt.title("Prediction: " + predicted_class_name.title())


import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(
    handle_base)
IMAGE_SIZE = (pixels, pixels)
# print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32

data_dir = tf.keras.utils.get_file(
    'people',
    'https://github.com/rikkhanna/keras-opencv-face-recog-api/blob/master/images.tgz',
    untar=True)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs)
else:
    train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)
