# Standart module
import pickle
from random import shuffle

# 3rd party module
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

# self defined module
from keras_ssd.ssd import SSD300
from keras_ssd.ssd_training import MultiboxLoss
from keras_ssd.ssd_utils import BBoxUtility

def load_data(file_path='VOC2007.pkl'):
    """function for using data from pkl file"""
    gt = pickle.load(open(file_path, 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]

    return train_keys, val_keys

def schedule(epoch, decay=0.9):
    """function for LearningRateScheduler"""
    return base_lr * decay**(epoch)

# training config
base_lr = 3e-4
nb_epoch = 100

# program config
np.set_printoptions(suppress=True)
path_prefix = './data/VOCdevkit/VOC2007/JPEGImages/'

# Constants
NUM_CLASSES = 4
input_shape = (300, 300, 3)

# SSD config
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# Dataload
train_keys, val_keys = load_data()

# Data generator
gen = Generator(gt, bbox_util, 16, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)

# make model
model = SSD300(input_shape, num_classes=NUM_CLASSES)

weights_file_path = 'weights_SSD300.hdf5'
if os.path.exists(weights_file_path):
    model.load_weights(weights_file_path, by_name=True)

## Not Trainable layer settings
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False

## Callback Settings for keras
callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]

optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)
