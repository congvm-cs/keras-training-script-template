import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import Sequence
import os
from config import *
import cv2


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, 
                dim=(32, 32), n_channels=1,
                isTraining = False,
                n_classes=10, 
                shuffle=True, 
                preprocessing_fn=None,
                featurewise_center=False, 
                samplewise_center=False, 
                featurewise_std_normalization=False, 
                samplewise_std_normalization=False, 
                zca_whitening=False, 
                zca_epsilon=1e-6, 
                rotation_range=0, 
                width_shift_range=0., 
                height_shift_range=0., 
                brightness_range=None, 
                shear_range=0., 
                zoom_range=0., 
                channel_shift_range=0., 
                fill_mode='nearest', 
                cval=0., 
                horizontal_flip=False, 
                vertical_flip=False, 
                rescale=None):
        # super(DataGenerator, self).__init__()

        # Image Data Generator
        self.data_augmentation = ImageDataGenerator(featurewise_center=featurewise_center, 
                                                    samplewise_center=samplewise_center, 
                                                    featurewise_std_normalization=featurewise_std_normalization, 
                                                    samplewise_std_normalization=samplewise_std_normalization, 
                                                    zca_whitening=zca_whitening, 
                                                    zca_epsilon=zca_epsilon, 
                                                    rotation_range=rotation_range, 
                                                    width_shift_range=width_shift_range, 
                                                    height_shift_range=height_shift_range, 
                                                    brightness_range=brightness_range, 
                                                    shear_range=shear_range, 
                                                    zoom_range=zoom_range, 
                                                    channel_shift_range=channel_shift_range, 
                                                    fill_mode=fill_mode, 
                                                    cval=cval, 
                                                    horizontal_flip=horizontal_flip, 
                                                    vertical_flip=vertical_flip)
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing_fn = preprocessing_fn
        self.isTraining = isTraining
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Get index from shuffled 

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) ## <------- should use (dim) (channel)
        y1 = np.empty((self.batch_size), dtype=np.float)
        y2 = np.empty((self.batch_size), dtype=np.float)

        # Generate data
        for i, idx in enumerate(indexes):
            # Store sample
            fname = self.list_IDs[idx][2::]
            img_path = os.path.join(ROOT_IMAGE_DIR, fname)
            img = img_to_array(load_img(img_path))
            # cv2.imwrite('test1.jpg', img)
            if self.preprocessing_fn is not None:
                img = self.preprocessing_fn(img)
            # cv2.imwrite('test2.jpg', img)
            # print('Done')
            if self.isTraining:
                img = self.data_augmentation.random_transform(img)
            # Store class
            X[i, ] = img
            y1[i] = self.labels[idx].split(' ')[0] # Age
            y2[i] = self.labels[idx].split(' ')[1] # Gender
        # print(y1)
        # print(y2)
        return X, [y1, y2]