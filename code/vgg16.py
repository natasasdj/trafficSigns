import os
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from sklearn.metrics import classification_report, confusion_matrix
#from utils import plot_training_history
#from utils import plot_confusion_matrix
#from utils import MeanPerClassAccuracy
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

             
# create instances of ImageDataGenerator for train and validation data

if args.data_augm:
    print("Data Augmentation")
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

img_size = [32, 32]
batch_size = 256

train_generator = gen_train.flow_from_directory(
  train_path,
  shuffle=True,
  target_size=img_size,
  batch_size=batch_size,
)

valid_generator = gen_valid.flow_from_directory(
  valid_path,
  shuffle = False,
  target_size=img_size,
  batch_size=batch_size,
)


# Model

num_classes = len(glob(train_path+'/*')) 
input_shape = img_size + [3]    
base_model = tf.keras.applications.VGG16(input_shape = input_shape, include_top=False, weights=None)                                              
x = Flatten()(base_model.output)
#x = GlobalAveragePooling2D()(base_model.output) 
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# compile model

model.compile(optimizer ='adam',
              loss ='categorical_crossentropy',
              metrics = ['accuracy']              
)
              
'''
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', MeanPerClassAccuracy(num_classes)],              
              #metrics=['accuracy', tf.keras.metrics.Recall(0), tf.keras.metrics.Recall(1)]
              )
'''

# Fit model

no_train_images = len(glob(train_path + '/*/*.ppm'))
no_valid_images = len(glob(valid_path + '/*/*.ppm'))

# create class_weght (imbalanced data: pay attention to underrepresented classes)
       
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

# fitting

epochs = 30
               
r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  class_weight = class_weight,
  steps_per_epoch = no_train_images / batch_size, 
  validation_steps = no_valid_images / batch_size
)


# save model history

import pandas as pd
import os
df  = pd.DataFrame(r.history)
fname = "vgg16.csv"
if os.path.exists(fname):
    df.to_csv(fname, index = False, mode = 'a', header = False)
else:
    df.to_csv(fname, index = False)
    
    
#save model

model_name = 'vgg16_e100'
#modelDir = '/home/natasa/share/trafficSigns/models'
modelDir = '.'
model.save(os.path.join(modelDir, model_name))

    
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

'''
r.history['loss']+=r1.history['loss']
r.history['val_loss'] += r1.history['val_loss']
r.history['accuracy'] += r1.history['accuracy']
r.history['val_accuracy'] += r1.history['val_accuracy']

r.history['mean_per_class_accuracy'] += r1.history['mean_per_class_accuracy']
r.history['val_mean_per_class_accuracy'] += r1.history['val_mean_per_class_accuracy']

'''


