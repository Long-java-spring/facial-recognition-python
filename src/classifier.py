"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import align.detect_face
import os
import sys
import math
import pickle
import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import Input, concatenate, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, add, Lambda
import tensorflow as tf
from keras import backend as K
from keras.utils.np_utils import to_categorical
from scipy import misc

EPOCHS = 300
INPUT_DIM = 512

def load_facenet_model(model_path, graph):
    with graph.as_default():
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(model_path)

def align_img(image, pnet, rnet, onet):
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor    
    face_crop_size = 160
    face_crop_margin = 32
    bounding_boxes, _ = align.detect_face.detect_face(image, minsize,
                                                          pnet, rnet, onet,
                                                          threshold, factor)
    img_size = np.asarray(image.shape)[0:2]
    for face in bounding_boxes:
        if face[4] > 0.50:
            det = np.squeeze(face[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - face_crop_margin / 2, 0)
            bb[1] = np.maximum(det[1] - face_crop_margin / 2, 0)
            bb[2] = np.minimum(det[2] + face_crop_margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + face_crop_margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            resized = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')
            return resized

def Model(clsno):
    model = keras.Sequential()
    model.add(Dense(2048, input_dim=INPUT_DIM, kernel_initializer="uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(clsno, kernel_initializer="uniform"))
    model.add(Activation('softmax'))
    return model

def get_y_true(llabel, clsno):
    y_true = []
    for label in llabel:
        y_true.append(to_categorical(label, num_classes=clsno))
    return np.array(y_true)

def train(class_names, emb_array, labels, batch_size, classifier_filename_exp):
    # Train classifier    
    clsno = len(class_names)
    print('Training classifier', clsno)
    model = Model(clsno)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    
    # with open('./models/lfw_classifier.pkl', 'rb') as infile:
    #     (model, _) = pickle.load(infile)
    
    model.fit(x=emb_array, y=get_y_true(labels, clsno), batch_size=batch_size, epochs=EPOCHS)    

    # Saving classifier model
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)

def predict(emb_array, labels, model, class_names):
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    
    # idx = 0
    for i in range(len(best_class_indices)):
        # if (best_class_probabilities[idx] < best_class_probabilities[i]):
        #     idx = i
        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

    if labels:    
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)
    
    return class_names[best_class_indices[0]], best_class_probabilities[0]
