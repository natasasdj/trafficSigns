import os
import itertools
import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_training_history
from utils import plot_confusion_matrix
from utils import MeanPerClassAccuracy
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-cw', '--class_weight', action='store_true')
parser.add_argument('-da','--data_augm', action='store_true')
args = parser.parse_args()

args.class_weight = True
if args.class_weight: print("Class Weight")

if args.data_augm: print("Data Augmentation")



model_name = '01_vanilla'
dataDir = '/home/natasa/share/trafficSigns/data'
train_path = os.path.join(dataDir, 'train')
valid_path = os.path.join(dataDir, 'validation')

# Build the model using the functional API

# Number of classes
num_classes = len(glob(train_path+'/*'))
print(num_classes)
# Image size
img_size = [32, 32]
input_shape = img_size + [3]
# Model
i = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(i, x)

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', MeanPerClassAccuracy(num_classes)])
              

# create instances of ImageDataGenerator for train and validation data
# TODO: Contrast Limited Adaptive Histogram Equalization 
'''
Data Augmentation
- brightness
- translating of image
- rotating of image
- Shearing the image
- Zooming in/out of the image
- Rather than generating and saving such images to hard disk, we generate them on the fly during training. 
'''
if args.data_augm:
    gen_train = ImageDataGenerator(rescale=1./255, 
                    brightness_range=[0.2,1.0], 
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2,
                    shear_range=0.1,
                    rotation_range=10.)
else:
    gen_train = ImageDataGenerator(rescale=1./255)
    

gen_valid = ImageDataGenerator(rescale=1./255)

# create generators

batch_size = 256

train_generator = gen_train.flow_from_directory(
  train_path,
  shuffle=True,
  target_size=img_size,
  batch_size=batch_size,
)

valid_generator = gen_valid.flow_from_directory(
  valid_path,
  target_size=img_size,
  batch_size=batch_size,
)


# create class_weght (imbalanced data: pay attention to underrepresented classes)
train_image_files = glob(train_path + '/*/*.ppm')
valid_image_files = glob(valid_path + '/*/*.ppm')
         
if args.class_weight:  
    print("Make Class Weght")
    total = len(train_image_files) 
    class_weight =  dict()
    for i in range(num_classes):
        no_per_class = len(glob(os.path.join(train_path,str(i),'*')) )
        if no_per_class>0:          
            class_weight[i] = (1/no_per_class) * total / num_classes
        else:
            class_weight[i]=1
    model_name += '_cw' 
else:
    class_weight = None              
    
                

epochs = 30
               
r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  class_weight = class_weight,
  steps_per_epoch=len(train_image_files) / batch_size, #steps_per_epoch,
  validation_steps=len(valid_image_files) / batch_size #, #validation_steps
  #verbose = 0
)

print("Model name", model_name)
modelDir = '/home/natasa/share/trafficSigns/models'
model.save(os.path.join(modelDir, model_name))

figDir = '/home/natasa/share/trafficSigns/figs'
plot_training_history(r, os.path.join(figDir, model_name))

gen_pred = ImageDataGenerator(rescale=1./255)
pred_generator = gen_pred.flow_from_directory(
  valid_path,
  shuffle = False,
  target_size=img_size,
  batch_size=batch_size,
)

Y_pred = model.predict(pred_generator, steps = len(valid_image_files) / batch_size, verbose = 1)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(pred_generator.classes, y_pred)
plot_confusion_matrix(cm, list(range(num_classes)), fname = os.path.join(figDir, model_name + '_confmatrix'))

import pandas as pd
pd.DataFrame({'loss':r.history['loss'][-1], 
            'accuracy': r.history['accuracy'][-1], 
            'mean_per_class_accuracy':r.history['mean_per_class_accuracy'][-1],
            

