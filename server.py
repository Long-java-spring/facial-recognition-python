from datetime import datetime
import sys
sys.path.append('./src')
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import numpy as np
import cv2
import os
import io
import classifier
import facenet
import align.detect_face
import re
import tensorflow as tf
import pickle
import boto3
from scipy import misc

app = Flask(__name__)
CORS(app, support_credentials=True)

BUCKET_NAME = 'facial-recognition-detected'
TRAINED_MODEL_KEY = 'autoface_classifier.pkl'
EMBEDDED_KEY = 'embedded.npy'
LABELS_KEY = 'labels.npy'

BATCH_SIZE=1000

global graph
global sess

model = None
class_names = []

graph = tf.get_default_graph()
sess = tf.Session()

classifier.load_facenet_model('./models/20180402-114759/20180402-114759.pb', graph)

# Get input and output tensors
images_placeholder = graph.get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

s3client = boto3.client(
    's3',
    'ap-northeast-1',
    aws_access_key_id='',
    aws_secret_access_key=''
)


try:    
    with open(EMBEDDED_KEY, 'wb') as data:
        s3client.download_fileobj(BUCKET_NAME, EMBEDDED_KEY, data)
    with open(LABELS_KEY, 'wb') as data:
        s3client.download_fileobj(BUCKET_NAME, LABELS_KEY, data)    
except:
    try:        
       os.remove(EMBEDDED_KEY)
       os.remove(LABELS_KEY)
    except:    
       pass # file does not exist
    pass
try:
    with open(TRAINED_MODEL_KEY, 'wb') as data:
        s3client.download_fileobj(BUCKET_NAME, TRAINED_MODEL_KEY, data)
    with open(TRAINED_MODEL_KEY, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
except:
    pass

pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

@app.route('/train', methods=['POST'])
def train():
    batch_size = BATCH_SIZE
    
    embeds = np.load(EMBEDDED_KEY).reshape(-1, classifier.INPUT_DIM)
    clsdata = np.load(LABELS_KEY)

    clsnames = list(set(clsdata))
    # clsnames = sorted(clsnames)
    labels = []
    for index, val in enumerate(clsdata):
        labels.append(clsnames.index(val))

    with graph.as_default():        
        print("len labels", len(clsnames))
        print("len data", len(embeds))

        classifier.train(clsnames, embeds, labels, batch_size, TRAINED_MODEL_KEY)
        s3client.upload_file(TRAINED_MODEL_KEY, BUCKET_NAME, TRAINED_MODEL_KEY)

        with open(TRAINED_MODEL_KEY, 'rb') as infile:
            global model
            global class_names
            (model, class_names) = pickle.load(infile)

        return jsonify("OK")

@app.route('/verify', methods=['POST'])
def verify():
    fdict = []
    data = request.json
    seconds_time = datetime.now().strftime("%m%d%Y%H%M%S")
    image = ''
    for b64img in data["images"]:
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', b64img)
            decoded_data = base64.b64decode(base64_data)
            np_data = np.fromstring(decoded_data, np.uint8)
            img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = classifier.align_img(img, pnet, rnet, onet)  
            image = img          
            # cv2.imwrite('verify.png', img)
            img = facenet.prewhiten(img)
            fdict.append(img)
        except Exception as e:
            return 'FAIL: an error has occurred %s'%str(e)

    with graph.as_default():
        feed_dict = {images_placeholder: fdict, phase_train_placeholder: False}
        predicts = sess.run(embeddings, feed_dict=feed_dict)
        predicts = predicts.reshape(len(fdict), 512)
        clsname, prob = classifier.predict(predicts, None, model, class_names)
        cv2.imwrite('%s_%s_%s.png'%(clsname, seconds_time, "%f"%prob), image)
        try:
            imgfn = '%s_%s_%s.png'%(clsname, seconds_time, "%f"%prob)
            s3client.upload_file(imgfn, BUCKET_NAME, "images/verify/%s"%imgfn)
            os.remove(imgfn)
        except:
            pass
        return jsonify({ "partId": clsname, "probability": "%f"%prob})

@app.route('/register', methods=['POST'])
def setup():
    data = request.json
    user_id = data["userId"]
    user_name = data["userName"]
    seconds_time = datetime.now().strftime("%m%d%Y%H%M%S")
    prewhiten_arr = []
    llabel = np.array([])

    if os.path.isfile(LABELS_KEY):
        llabel = np.load(LABELS_KEY)

    for cnt, b64img in enumerate(data["images"]):
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', b64img)
            decoded_data = base64.b64decode(base64_data)
            np_data = np.fromstring(decoded_data, np.uint8)
            img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
            if img.ndim == 0:
                img = facenet.to_rgb(img)
            img = classifier.align_img(img, pnet, rnet, onet)
            cv2.imwrite('%s_%s_%s.png'%(seconds_time, user_name, cnt + 1), img)
            img = facenet.prewhiten(img)
            prewhiten_arr.append(img)
            llabel = np.append(llabel, '%s'%user_id)
        except Exception as e:
            print('Error: ', str(e))
            pass

    with graph.as_default():
        feed_dict = {images_placeholder: prewhiten_arr, phase_train_placeholder: False}
        embedd_data = sess.run(embeddings, feed_dict=feed_dict)
        employee_emb = '%s.npy'%(user_id)
        np.save(employee_emb, embedd_data)
        s3client.upload_file(employee_emb, BUCKET_NAME, 'embedded/%s'%employee_emb)
    
        if os.path.isfile(EMBEDDED_KEY):
            embed = np.load(EMBEDDED_KEY).reshape(-1, classifier.INPUT_DIM)
            embed = np.vstack((embed, embedd_data))
            if llabel.size * classifier.INPUT_DIM != embed.size:
                return "FAIL: dim label and embedded is not the same"
        else:
            embed = embedd_data
        np.save(EMBEDDED_KEY, embed)
        s3client.upload_file(EMBEDDED_KEY, BUCKET_NAME, EMBEDDED_KEY)

        np.save(LABELS_KEY, llabel)
        s3client.upload_file(LABELS_KEY, BUCKET_NAME, LABELS_KEY)
        return "OK"

@app.route('/delete', methods=['POST'])
def delete():
    data = request.json
    employee_id = data["userId"]
    labels = np.array([])
    embed = np.array([]).reshape(-1, classifier.INPUT_DIM)

    embedding_key = np.where(labels == employee_id)
    for row in embedding_key[0]:
        embed = np.delete(embed, row, 0)

    np.save(EMBEDDED_KEY, embed)
    s3client.upload_file(EMBEDDED_KEY, BUCKET_NAME, EMBEDDED_KEY)

    labels = labels[labels != employee_id]
    np.save(LABELS_KEY, labels)
    s3client.upload_file(LABELS_KEY, BUCKET_NAME, LABELS_KEY)

    return "OK"
if __name__ == '__main__':
   app.run(host="0.0.0.0", port="5000", threaded=True)
