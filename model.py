import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, BatchNormalization, GlobalMaxPooling2D, Dense
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Activation, Input
# from keras_applications.inception_v3 import InceptionV3

class ZAGNet:
    def __init__(self, input_shape, dropout=0.2, alpha=1):
        self.input_shape=input_shape
        self.alpha = alpha
        self.after_pooling_shape = (1, 1, 1024)
        self.dropout = dropout


    def __call__(self):
        inputs = Input(shape=self.input_shape)
        #-------------------------------------------------------------------------------#
        base_model = MobileNet(input_shape=self.input_shape, include_top=False, alpha=1)
        # print(base_model.summary())
        x = base_model(inputs)

        #-------------------------------------------------------------------------------#
        x = GlobalAveragePooling2D(name='GlobleAvPool')(x)
        x = Reshape(self.after_pooling_shape, name='reshape1')(x)   # 1x1x2048
        # x = Conv2D(2048, (3, 3), padding='same', name='shared_conv')
        x = Dropout(self.dropout, name='dropout1')(x)
        x = BatchNormalization(name='batch_norm1')(x)
        
        #-------------------------------------------------------------------------------#
        age_emb = Conv2D(1, (1, 1), name='age_emb',padding='same', activation='linear')(x)
        age_out = Reshape((1,), name='age_out')(age_emb)
        
        #-------------------------------------------------------------------------------#
        gender_emb = Conv2D(1, (1, 1), name='gender_emb', padding='same', activation='sigmoid')(x)
        gender_out = Reshape((1,), name='gender_out')(gender_emb)

        #-------------------------------------------------------------------------------#
        final_model = Model(inputs=inputs, outputs=[age_out, gender_out])
        

        # Loading weights
        # weight_name = './model_ckps/age-only-lr-ml-ZAGNET-ft-1-bm-Inception_V3-is-160-dr-0.2-bs-64-lr-0.05-epoch-05-val_loss-4.4824-val_age_loss-12.6144-val_gender_loss-0.6980.h5df' 
        # final_model.load_weights(weight_name, by_name=True, skip_mismatch=True)
        # print('loaded weights: {}'.format(weight_name))
        
        # print('---------------------------------------------------')
        # print('Freezing all layers except: ')
        return final_model