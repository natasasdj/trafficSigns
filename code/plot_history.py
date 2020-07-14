import pandas as pd
import matplotlib.pyplot as plt

model = 'vgg16'
fname = '/home/natasa/share/trafficSigns/results/' + model + '.csv'
df1 = pd.read_csv(fname)


df = df.iloc[0:180,:]

fname = '/home/natasa/share/trafficSigns/figs/' + model
plt.figure(1)
plt.title('Loss')
plt.plot(df['loss'], 'r', label='loss', )
plt.plot(df['val_loss'], 'b', label='val_loss')
#plt.ylim([-0.1,3.0])
plt.legend()
plt.grid()
plt.savefig(fname + '_loss.png')

plt.figure(2)
plt.title('Accuracy')
plt.plot(df['accuracy'], 'r-', label='acc')
plt.plot(df['val_accuracy'],  'b-', label='val_acc')
#plt.ylim([0.5,1.01])
plt.legend()
plt.grid()
plt.savefig(fname + '_acc.png')
plt.show()

