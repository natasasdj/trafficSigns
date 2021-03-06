import os
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from sklearn.metrics import classification_report, confusion_matrix
#from utils import plot_training_history
#from utils import plot_confusion_matrix
from utils import MeanPerClassAccuracy
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-cw', '--class_weight', action='store_true')
parser.add_argument('-da','--data_augm', action='store_true')
args = parser.parse_args()

args.class_weight = True
args.data_augm = True

mainDir = '/home/natasa_sarafijanovic'
dataDir = os.path.join(mainDir,'data')
train_path = os.path.join(dataDir, 'train')
valid_path = os.path.join(dataDir, 'validation')


# Number of classes
num_classes = len(glob(train_path+'/*'))
print(num_classes)
# Image size
img_size = [32, 32]
              

# create instances of ImageDataGenerator for train and validation data

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


# Model

input_shape = img_size + [3]

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
              metrics=['accuracy'])

# create class_weght (imbalanced data: pay attention to underrepresented classes)

no_train_images = len(glob(train_path + '/*/*.ppm'))
no_valid_images = len(glob(valid_path + '/*/*.ppm'))

       
if args.class_weight:
    total = no_train_images 
    class_weight =  dict()
    for i in range(num_classes):
        no_per_class = len(glob(os.path.join(train_path,str(i),'*')) )
        if no_per_class>0:          
            class_weight[i] = (1/no_per_class) * total / num_classes
        else:
            class_weight[i]=1    
else:
    class_weight = None              

# fit model

epochs = 20

r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  class_weight = class_weight,
  steps_per_epoch = no_train_images / batch_size, 
  validation_steps = no_valid_images / batch_size
)

# save model history
df  = pd.DataFrame(r.history)
fname = "vgg6conv.csv"
if os.path.exists(fname):
    df.to_csv(fname, index = False, mode = 'a', header = False)
else:
    df.to_csv(fname, index = False)
    
# save model
model_name = 'vgg6conv_e70'
#modelDir = '/home/natasa/share/trafficSigns/models'
modelDir = '.'
model.save(os.path.join(modelDir, model_name))


#figDir = '/home/natasa/share/trafficSigns/figs'
#plot_training_history(r, os.path.join(figDir, model_name))

# Evaluation on the validation dataset

test_path = valid_path
no_test_images = no_valid_images

gen_test = ImageDataGenerator(rescale=1./255)
test_generator = gen_test.flow_from_directory(
  test_path,
  shuffle = False,
  target_size=img_size,
  batch_size=batch_size)

Y_pred = model.predict(test_generator, steps = no_test_images / batch_size, verbose = 1)

y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)
cr = classification_report(test_generator.classes, y_pred, digits = 4)
print(cm)
print(cr)

