import os
import tensorflow as tf
import keras
from keras.applications import mobilenet
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import numpy as np
from config import *
from model import ZAGNet
from transform import zalo_preprocessing_function, preprocessing_function
from utils import lrSched
from generator_sequence import DataGenerator
from custom_callbacks import CyclicLR, CSVLogger_Iteration


# ##############################################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Force to use CPU

# Takeover X% GPU Memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))


# ##############################################################################
# Load Dataset
X_train, Y_train = np.load('./npy_data/megaasian_age_gender_train.npy')
X_val, Y_val = np.load('./npy_data/megaasian_age_gender_test.npy')
print('Numbers of X_train: {}'.format(len(X_train[-1])))
print('Numbers of X_val: {}'.format(len(X_val[-1])))

# ##############################################################################
# Data Generator
train_generator = DataGenerator(list_IDs=X_train[-1],
								labels=Y_train[-1], 
								dim=(INPUT_SIZE, INPUT_SIZE),
								n_channels=3,
								batch_size=BATCH_SIZE, 
								preprocessing_fn=preprocessing_function,
								shuffle=True,
								shear_range=0.3,
								zoom_range=0.3,
								horizontal_flip=True, 
								rotation_range=30, 
								width_shift_range=0.2, 
								height_shift_range=0.2,
								isTraining=True)

val_generator = DataGenerator(list_IDs=X_val[-1],
								labels=Y_val[-1], 
								dim=(INPUT_SIZE, INPUT_SIZE),
								n_channels=3,
								batch_size=BATCH_SIZE, 
								preprocessing_fn=preprocessing_function,
								shuffle=False, 
								isTraining=False)

# ##############################################################################
# Load Model
# model = ZAGNet(input_shape=INPUT_SHAPE, dropout=DROPOUT)()
weights_fname = './model_ckps/ml-ZAGNET-ft-1-bm-MobileNet-is-128-dr-0.2-bs-32-lr-0.05-aw-0.5-gw-1-epoch-100-val_loss-1.8904-val_age_loss-3.2396-val_gender_loss-0.2706-val_gender_out_binary_accuracy-0.9019.h5'
model = load_model(weights_fname, custom_objects={'relu6': mobilenet.relu6})
print(model.summary())
print('loaded weights: ', weights_fname)
print('-----------------------------------------------------')


# Freezing layers
print('All layers is freezed except:')
for layer in model.layers:
	layer.trainable = False
	# print(layer.name)

# Custom model
print('# Customized Addition Layers')
for layer in model.layers[::-1][:9]:
	layer.trainable = True
	print('> ', layer.name)

# MobileNet model
print('\n')
# print('#MobileNet Model')
# for layer in model.layers[1].layers[::-1][:6]:
# 	layer.trainable = True
# 	print('> ',layer.name)
print('---------------------------------------------------')


# ##############################################################################
# Load Callbacks
CSV_FNAMES = './CSVlogs/ml-{}-ft-{}-bm-{}-is-{}-dr-{}-bs-{}-lr-{}-aw-{}-gw-{}.csv'.format(
			MODEL_NAME,
			FINETUNING_VERSION,
			BASED_MODEL_NAME,
			INPUT_SIZE,
			DROPOUT, 
			BATCH_SIZE,
			LEARNING_RATE,
			AGE_LOSS_WEIGHT,
			GENDER_LOSS_WEIGHT)

fname = './model_ckps/ml-{}-ft-{}-bm-{}-is-{}-dr-{}-bs-{}-lr-{}-aw-{}-gw-{}'.format(MODEL_NAME,
														FINETUNING_VERSION,
														BASED_MODEL_NAME,
														INPUT_SIZE,
														DROPOUT, 
														BATCH_SIZE,
														LEARNING_RATE,
														AGE_LOSS_WEIGHT,
														GENDER_LOSS_WEIGHT)

WEIGHT_CKP = fname + '-epoch-{epoch:02d}-val_loss-{val_loss:.4f}-val_age_loss-{val_age_out_loss:.4f}-val_gender_loss-{val_gender_out_loss:.4f}-val_gender_out_binary_accuracy-{val_gender_out_binary_accuracy:.4f}.h5'


csvlogger_callback = CSVLogger(CSV_FNAMES, append=True)

csvlogger_iter_callback = CSVLogger_Iteration('CSV_FNAMES.csv', append=True)

model_cp_callback = ModelCheckpoint(WEIGHT_CKP, 
									monitor='val_loss',
									verbose=1,
									save_weights_only=False,
									mode='min', 
									save_best_only=False)

lr_callback = LearningRateScheduler(lrSched, verbose=1)
tensorboard = TensorBoard(log_dir='./TBlogs')
cyclr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=1000.)


callback_list = [csvlogger_callback, model_cp_callback, cyclr, csvlogger_iter_callback]


# ##############################################################################
# Compile 
opt = SGD(LEARNING_RATE, momentum=0.95)
model.compile(optimizer=opt, 
				loss={
				'age_out': 'mae',
				'gender_out': 'binary_crossentropy',
				}, 
				metrics={
				'age_out': 'mae',
				'gender_out': 'binary_accuracy'
				},
				loss_weights={
				'age_out': AGE_LOSS_WEIGHT,
				'gender_out': GENDER_LOSS_WEIGHT
				})


# Compute sample weights
from sklearn.utils.class_weight import compute_sample_weight
clc = []
for i in Y_train[-1]:
	clc.append(i.split(' ')[0])

sample_weighted = compute_sample_weight(class_weight='balanced', y=clc)


# ##############################################################################
# Train
model.fit_generator(train_generator,
			validation_data=val_generator,
			epochs=EPOCHS,
			verbose=1, 
			use_multiprocessing=True,
			workers=64,
			initial_epoch=INITIAL_EPOCH, 
			callbacks = callback_list,
			class_weight={
				'age_out': sample_weighted
			},
			max_queue_size=len(X_train[-1]))

