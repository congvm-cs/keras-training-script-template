# Generator
from keras.preprocessing.image import load_img, img_to_array
import keras
from keras.utils import to_categorical
import numpy as np
import cv2
import os 
from config import *


def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    w = x.shape[0]
    h = x.shape[1]
    out = x[0+dx:w-(dn-dx),0+dy:h-(dn-dy),:]
    out = cv2.resize(out, (w, h))
    return out


def augment_data(images):
    for i in range(0, images.shape[0]):
        if np.random.random() > 0.5:
            images[i] = images[i][:,::-1]

        if np.random.random() > 0.5:
            images[i] = random_crop(images[i], 4)
        
        if np.random.random() > 0.75:
            images[i] = keras.preprocessing.image.random_rotation(images[i], 20, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = keras.preprocessing.image.random_shear(images[i], 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = keras.preprocessing.image.random_shift(images[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = keras.preprocessing.image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)    
    return images


def data_generator_reg(X, Y, batch_size, preprocessing_fn, image_size=128, shuffle=False, image_dir=None):
    X = np.array(X, dtype=str)
    Y = np.array(Y)
    while True:
        if shuffle == True:
            idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p, q = [],[]
        for i in range(len(X)):
            img_path = None
            if image_dir is not None:
                img_path = os.path.join(ROOT_IMAGE_DIR, X[i])

            img = img_to_array(load_img(img_path))
            img = preprocessing_fn(img)
            p.append(img)
            [age, gender] = Y[i].split(' ')
            q.append([int(age), int(gender)])

            print(q)
            if len(p) == batch_size:
                yield augment_data(np.array(p)), np.array(q)
                p,q = [],[]