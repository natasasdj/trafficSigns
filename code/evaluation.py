import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_confusion_matrix
from utils import MeanPerClassAccuracy, MeanPerClassAccuracy2
from glob import glob

#dataDir = '/home/natasa/share/trafficSigns/data'
dataDir = '/home/natasa_sarafijanovic/data'
test_path = os.path.join(dataDir, 'test')
no_test_images = len(glob(test_path + '/*/*.ppm'))
img_size = (32,32)
batch_size =256


model = tf.keras.models.load_model('/home/natasa_sarafijanovic/vgg6conv_e70')
#model = tf.keras.models.load_model('/home/natasa/share/trafficSigns/models_/01_vanilla_e50', custom_objects = {'MeanPerClassAccuracy':MeanPerClassAccuracy2})

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


with open('cm', 'wb') as f:
    pickle.dump(cm, f)

fname = '/home/natasa/share/trafficSigns/results/cm'
with open(fname, 'rb') as f:
    cm = pickle.load(f)

model_name ='vgg6conv_e70'
figDir =  '/home/natasa/share/trafficSigns/figs' 
plot_confusion_matrix(cm, list(range(num_classes)), fname = os.path.join(figDir, model_name + '_confmatrix'))   
