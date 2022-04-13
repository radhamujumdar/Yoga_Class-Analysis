import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import pandas as pd
import csv
import time
import os
import numpy as np
import cv2
from glob import glob
from posemodule import*
import posemodule as pm


# Import matplotlib libraries
from matplotlib import pyplot as plt

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")
def save_frame(video_path, gap=30):

    save_path="frames"
    create_dir(save_path)
#     print("yes")

    cap = cv2.VideoCapture(video_path)
#     print("yes again")
    #fps = cap.get( cv2.CAP_PROP_FPS )
    #print(fps)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if (idx>=15 and (idx-gap//2) % gap == 0):
            #image=img_crop(frame)
            cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1
if __name__ == "__main__":
    video_paths = glob("testing/*")
    print(video_paths)
#     save_dir = "geet_dataset2/train/"
    for path in video_paths:
        save_frame(path, gap=60)

def image_to_movenet(path):
    image_path =path
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # Download the model from TF Hub.
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']

    # Run model inference.
    outputs = movenet(image)
# Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    kpts_x = keypoints[0, 0, :, 0]
    kpts_y = keypoints[0, 0, :, 1]
    kpts_scores = keypoints[0, 0, :, 2]




     # Empty dataset
    t_x=kpts_x.numpy()
    t_y=kpts_y.numpy()

    tn=[]

    for i in range(0,17):
        tn.append(t_x[i])
        tn.append(t_y[i])
    return tn

path_lst=[]

keypts_inner_test=[]
keypts_outer_test=[]



for image in os.listdir('frames'):
    path_lst.append(os.path.abspath(f"frames/{image}"))

        #path_lst_per.append(path_lst)
        #image_path_dictionary[name] = path_list

path_lst = sorted(path_lst,key=os.path.getmtime)

for path in path_lst:
    print(path)
    keypts_inner_test=image_to_movenet(path)
    keypts_outer_test.append(keypts_inner_test)
print(keypts_outer_test)

#padding and masking
from keras.layers import Masking
def padding_masking(x_test):
    testing=[]
    for i in range(0,1):
        result = np.zeros((77, 34))
        result[:np.array(x_test[0]).shape[0],:np.array(x_test[0]).shape[0]]=x_test[0]
        testing.append(result)
    x_test=tf.convert_to_tensor(testing,dtype='float32')
    mask_layer=Masking(mask_value=0.0)
    x_test=mask_layer(x_test)
    return x_test
x_test=keypts_outer_test
x_test=np.array(x_test)
x_test.shape[0]
x_test=padding_masking(x_test)
print(x_test.shape)

from tensorflow import keras
# It can be used to reconstruct the model identically.
model = keras.models.load_model("savemodel.h5")
y_pred = model.predict(x_test)
print(np.argmax(y_pred))
if y_pred.any()>float(2):
    pass
else:
    print('Asana not out of the 6 specified')
    exit()
final_label=np.argmax(y_pred)

detector = pm.poseDetector()
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (1280, 720))]
ts = 0
found = None
for file_name in glob('frames/*'):
    fts = os.path.getmtime(file_name)
    if fts > ts:
        ts = fts
        found = file_name

print(found)
image = cv2.imread(found)
image=cv2.resize(image,(1280,720))
img = detector.findPose(image,False)
lmList = detector.findPosition(img, False)
print(lmList)
if len(lmList) != 0:
    if final_label==0:
        angle = detector.findAngle(img, 11,23,25,final_label)
    elif final_label==1:
        angle = detector.findAngle(img, 24,26,28,final_label)
        angle = detector.findAngle(img, 23,25,27,final_label)
    elif final_label==2:
        angle = detector.findAngle(img, 12,24,28,final_label)
    elif final_label==3:
        angle = detector.findAngle(img, 21,15,13,final_label)
        angle = detector.findAngle(img, 22,16,14,final_label)
    # angle2 = detector.findAngle(img, 13, 23, 27)
    elif final_label==4:
        angle = detector.findAngle(img, 14,24,28,final_label)
        angle = detector.findAngle(img, 13,23,27,final_label)
    elif final_label==5:
        angle = detector.findAngle(img, 24,26,28,final_label)
        angle = detector.findAngle(img, 23,25,27,final_label)



cv2.imshow("Image", img)
cv2.waitKey(0)
