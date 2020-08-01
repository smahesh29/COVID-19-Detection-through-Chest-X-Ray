# USAGE
# python corona.py --image image_path

import cv2
import numpy as np 
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def test(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = load_model('covid19.model')

    img = cv2.imread(img)
    img = cv2.resize(img,(224, 224))
    result_img = cv2.resize(img,(600, 600))
    #plt.imshow(img)
    img = np.reshape(img,[1,224,224,3])
    array = model.predict(img)
    result = array.argmax(axis=-1)
    print(array)
    print(result)
    if result[0] == 1:
        prediction = 'normal'
    else:
        prediction = 'covid'

    print("Result : ", prediction)
    if prediction == 'normal':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255) 
    cv2.putText(result_img,prediction,(25,25), font, 1, color, 2, cv2.LINE_AA)

    return result_img

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to testing x-ray image")
args = vars(ap.parse_args())

result_img = test(args["image"])
cv2.imshow("Result",result_img)
cv2.waitKey(0)

