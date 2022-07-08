import tensorflow as tf
from align import detect_face
import cv2
import imutils
import numpy as np
import argparse
import os
from scipy import misc
import base64
import requests
import json

parser = argparse.ArgumentParser()
parser.add_argument("--imgpath", type = str, required=True)
args = parser.parse_args()

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

def dotest(f):
    img = cv2.imread(os.path.join(args.imgpath, f))
    img = imutils.resize(img,width=1024)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                #resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_AREA)
                resized = misc.imresize(cropped, (input_image_size, input_image_size), interp='bilinear')
                misc.imsave("aligned_%s"%f, resized)
                postrequest("aligned_%s"%f)
    #return faces

def postrequest(file):
    url = 'http://34.245.28.101:5000/face'
    #url = 'http://localhost:5000/face'

    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        # Convert it to a readable utf-8 code (a String)
        encoded_string = encoded_string.decode('utf-8')

    data = {"imgs":[encoded_string]}
    #print(encoded_string)
    
    r = requests.post(url, auth=None, verify=False, json=data)
    print(file, r.status_code, r.content)

for f in os.listdir(args.imgpath):
    if f == '.' or f == '..':
        continue
    print("image ", f)
    dotest(f)
    
