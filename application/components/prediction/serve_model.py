from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = None


def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 2)[0]

    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"

        response.append(resp)

    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


import cv2
from glob import glob
from pathlib import Path

#from utils import plot
from fastapi import FastAPI, File, UploadFile

#import torch
#import warnings
#warnings.filterwarnings("ignore")

"""# Initilisation API"""



#from __future__ import absolute_import, division, print_function, unicode_literals

#import matplotlib.pylab as plt
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np

"""For better data visualization we'll use [Pandas library](https://pandas.pydata.org/)."""

#import pandas as pd

# Increase precision of presented data for better side-by-side comparison
#pd.set_option("display.precision", 8)


"""# Load TFLite model
Load TensorFlow lite model with interpreter interface.
"""


# Load TFLite model and see some details about input/output
#from __future__ import absolute_import, division, print_function, unicode_literals
#!wget https://s3.eu-central-1.amazonaws.com/lms-lyon.fr/modelSavedOptimized.tar
#!tar -xvf "/content/modelSavedOptimized.tar" -C "/content/"  
#import matplotlib.pylab as plt
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np


#tflite_interpreter = tf.lite.Interpreter(model_path="modelSavedOptimized.tflite")
#tflite_interpreter.allocate_tensors()

#input_details = tflite_interpreter.get_input_details()
#output_details = tflite_interpreter.get_output_details()

# print("== Input details ==")
# print("name:", input_details[0]['name'])
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])

# print("\n== Output details ==")
# print("name:", output_details[0]['name'])
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])




"""# Test"""

import numpy as np
from skimage.transform import resize
import cv2

def video_mamonreader(cv2,filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i=0
    print(frames.shape)
    vc = cv2.VideoCapture(filename)
    #vc = cv2.cvtColor(vc, cv2.COLOR_BGR2GRAY)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frm = resize(frame,(160,160,3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    print("reading video")
    while i < 30:
        rval, frame = vc.read()
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
        print(i)
    return frames



def pred_fight(video,acuracy=0.85):
  ysvid2 = video_mamonreader(cv2,video)
  ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float32)
  ysdatav2[0][:][:] = ysvid2
  tflite_interpreter.set_tensor(input_details[0]['index'], ysdatav2)
  tflite_interpreter.invoke()
  tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])    
    
  if tflite_model_predictions[0][1] >=acuracy:
      return True , tflite_model_predictions[0][1]
  else:
      return False , tflite_model_predictions[0][1]


def main_fight(vidoss):
    import time
    vid = video_mamonreader(cv2,vidoss)
    datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    datav[0][:][:] = vid
    millis = int(round(time.time() * 1000))
    #print(millis)
    f , precent = pred_fight(vidoss,acuracy=0.9)
    millis2 = int(round(time.time() * 1000))
    #print(millis2)
    res_mamon = {'fight':f , 'precentegeoffight':str(precent)}
    res_mamon['processing_time'] =  str(millis2-millis)
    return res_mamon
