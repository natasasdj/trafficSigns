# Plot confusion matrix
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_confusion_matrix(cm, classes, fname = None, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')
  print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  
  tick_marks_y = np.array(range(-1, len(classes)+2)) - 0.1
  tick_marks_x = np.array(classes)
  plt.xticks(tick_marks_x, classes, rotation=45)
  plt.yticks(tick_marks_y, [''] + classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i+0.2, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="magenta" if cm[i, j] > thresh else "black") 
                            
  plt.ylim([classes[-1]+0.5, classes[0]-0.5])
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  if fname: plt.savefig(fname+'.png')
  plt.show()



def plot_training_history(r, fname=None):
    plt.figure(1)
    plt.title('Loss')
    plt.plot(r.history['loss'], 'r', label='loss', )
    plt.plot(r.history['val_loss'], 'b', label='val_loss')
    plt.ylim([-0.1,1.0])
    plt.legend()
    if fname: plt.savefig(fname + '_loss.png')
    plt.figure(2)
    plt.title('Accuracy')
    plt.plot(r.history['accuracy'], 'r-', label='acc')
    plt.plot(r.history['val_accuracy'],  'b-', label='val_acc')
    plt.plot(r.history['mean_per_class_accuracy'], 'r--' , label='mean_per_class_acc' )
    plt.plot(r.history['val_mean_per_class_accuracy'], 'b--', label='val_mean_per_class_acc')
    plt.ylim([0.9,1.01])
    plt.legend()
    if fname: plt.savefig(fname + '_acc.png')
    plt.show()



'''
def plot_confusion_matrix(cm, classes, fname = None, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    normalize=False; title='Confusion matrix'; cmap=plt.cm.Blues
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_y = np.array(range(-1, len(classes)+2))
    tick_marks_x = np.array(classes)
    classes_y = [''] + classes
    plt.xticks(tick_marks_x, classes, rotation=45)
    plt.yticks(tick_marks_y, classes_y)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      _ = plt.text(j, i+ccc, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="orange" if cm[i, j] > thresh else "black")         
    plt.ylim([classes[-1]+0.5, classes[0]-0.5])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.ylim([-0.1, K])
    plt.tight_layout()
    if fname: plt.savefig(fname+'.png')
    plt.show()
'''

class MeanPerClassAccuracy(tf.keras.metrics.Recall):
    def __init__(self, num_classes, name="mean_per_class_accuracy", **kwargs):
        super(MeanPerClassAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.num_classes = num_classes
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.count.assign_add(1)
        acc = 0.
        for class_id in range(self.num_classes):
            self._reset_states()
            self.class_id = class_id
            super(MeanPerClassAccuracy, self).update_state(y_true, y_pred) 
            acc += super(MeanPerClassAccuracy, self).result()
        self.accuracy.assign_add(acc/self.num_classes)           
    def result(self):
        return self.accuracy/self.count
    def _reset_states(self):
        self.true_positives.assign(tf.zeros(shape=(len(self.thresholds),)))
        self.false_negatives.assign(tf.zeros(shape=(len(self.thresholds),)))
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self._reset_states()
        self.accuracy.assign(0.0)
        self.count.assign(0.0)

'''  
class History: 
    history = dict()

r = History()

r.history['loss'] = r1.history['loss'] + r2.history['loss'] + r3.history['loss']
r.history['val_loss'] = r1.history['val_loss'] + r2.history['val_loss'] + r3.history['val_loss']

r.history['loss'] = r1.history['loss'] + r2.history['loss'] + r3.history['loss']
r.history['accuracy'] = r1.history['accuracy'] + r2.history['accuracy'] + r3.history['accuracy']
r.history['val_accuracy'] = r1.history['val_accuracy'] + r2.history['val_accuracy'] + r3.history['val_accuracy']
r.history['mean_per_class_accuracy'] = r1.history['mean_per_class_accuracy'] + r2.history['mean_per_class_accuracy'] + r3.history['mean_per_class_accuracy']
r.history['val_mean_per_class_accuracy'] = r1.history['val_mean_per_class_accuracy'] + r2.history['val_mean_per_class_accuracy'] + r3.history['val_mean_per_class_accuracy']
  
'''



































