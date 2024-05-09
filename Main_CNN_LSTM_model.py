import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# with h5py.File('E:\SSEs\pack_output\data\cmemes.h5','r') as f9:
#     X9 = f9['X'][:]
#     Y9 = f9['Y'][:]
# with h5py.File('E:\SSEs\pack_output\data\jcope.h5','r') as f10:
#     X10 = f10['X'][:]
#     Y10 = f10['Y'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF01.h5','r') as f1:
    X1 = f1['X'][:]
    Y1 = f1['Y'][:]
    # val_in1 = f1['val_in'][:]
    # val_out1 = f1['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF02.h5','r') as f2:
    X2 = f2['X'][:]
    Y2 = f2['Y'][:]
    # val_in2 = f2['val_in'][:]
    # val_out2 = f2['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF03.h5','r') as f3:
    X3 = f3['X'][:]
    Y3 = f3['Y'][:]
    # val_in3 = f3['val_in'][:]
    # val_out3 = f3['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF04.h5','r') as f4:
    X4 = f4['X'][:]
    Y4 = f4['Y'][:]
    # val_in4 = f4['val_in'][:]
    # val_out4 = f4['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF05.h5','r') as f5:
    X5 = f5['X'][:]
    Y5 = f5['Y'][:]
    # val_in5 = f5['val_in'][:]
    # val_out5 = f5['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF06.h5','r') as f6:
    X6 = f6['X'][:]
    Y6 = f6['Y'][:]
    # val_in6 = f6['val_in'][:]
    # val_out6 = f6['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF07.h5','r') as f7:
    X7 = f7['X'][:]
    Y7 = f7['Y'][:]
    # val_in7 = f7['val_in'][:]
    # val_out7 = f7['val_out'][:]
with h5py.File('E:\SSEs\pack_output\data\hycomF08.h5','r') as f8:
    X8 = f8['X'][:]
    Y8 = f8['Y'][:]
    # val_in8 = f8['val_in'][:]
    # val_out8 = f8['val_out'][:]

data_X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8))
data_Y = np.concatenate((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8))
# val_in = np.concatenate((val_in1,val_in2,val_in3,val_in4,val_in5,val_in6,val_in7,val_in8))
# val_out = np.concatenate((val_out1,val_out2,val_out3,val_out4,val_out5,val_out6,val_out7,val_out8))

train_X_ = data_X.reshape(-1,60,1)
train_Y_ = data_Y.reshape(-1,60,1)
train_X = train_X_.astype('float32')
train_Y = train_Y_.astype('float32')
for iii in range((train_Y.shape[0])):
    train_Y[iii,:,0] = np.mean(train_Y[iii,:,0]) + 1.5*(train_Y[iii,:,0]-np.mean(train_Y[iii,:,0]))

from sklearn.model_selection import train_test_split

train_X,test_X,train_Y,test_Y = train_test_split(train_X, train_Y, test_size=0.15, random_state=15)
train_X,val_X,train_Y,val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=15)
# train_X,valid_X,train_Y,valid_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=15)
print(train_X.shape,test_X.shape,val_X.shape,train_Y.shape,test_Y.shape,val_Y.shape)
# print(train_X.shape,test_X.shape,train_Y.shape,test_Y.shape)

# Model the data
BATCH_SIZE = 128
EPOCHS = 400
num_classes = 90
# the neural network
import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
import tensorflow.keras.optimizers as optimizers

model_m = Sequential()
model_m.add(Conv1D(8, 9, padding='same', activation='selu', input_shape=(60, 1)))
# model_m.add(tf.keras.layers.BatchNormalization())
model_m.add(Conv1D(16, 9, padding='same', activation='selu'))
# model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(16, 7, padding='same', activation='selu'))
model_m.add(Conv1D(32, 7, padding='same', activation='selu'))

model_m.add(Conv1D(64, 5, padding='same', activation='selu'))
model_m.add(Conv1D(32, 5, padding='same', activation='selu'))
# model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(32, 3, padding='same', activation='selu'))
model_m.add(Conv1D(16, 3, padding='same', activation='selu'))
# model_m.add(MaxPooling1D(2))
# model_m.add(Dropout(rate=0.2))
model_m.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
model_m.add(Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.1)))
model_m.add(LSTM(units=8, return_sequences=True))
model_m.add(Dropout(rate=0.1))
model_m.add(Flatten())
model_m.add(Dense(train_Y.shape[1], activation='linear'))

# opt = optimizers.SGD(learning_rate=0.00001, momentum=0.9)
model_m.compile(loss='mean_squared_logarithmic_error',optimizer='adam', metrics=['mse'])
print(model_m.summary())
# train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=10)
model_m_train = model_m.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                            callbacks=[early_stopping_monitor],
                            validation_data=(test_X, test_Y))

plt.figure(1)
plt.plot(model_m_train.history['mse'])
plt.plot(model_m_train.history['val_mse'])
plt.title('Model mse')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.figure(2)
plt.plot(model_m_train.history['loss'])
plt.plot(model_m_train.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

_, train_mse = model_m.evaluate(train_X, train_Y, verbose=0)
_, test_eval = model_m.evaluate(test_X, test_Y, verbose=0)
print('Test loss: ', train_mse)
print('Test accuracy: ', test_eval)

# precision = (model_m_train.history['precision'])
# val_precision = model_m_train.history['val_precision']
loss = model_m_train.history['loss']
val_loss = model_m_train.history['val_loss']

history=model_m_train.history
import csv
w = csv.writer(open('E:\SSEs\pack_output\data\LossHistory.csv', "w"))
for key, val in history.items():
    w.writerow([key, val])
# save model
# serialize model to YAML
# from tensorflow.keras.models import model_from_yaml
model_yaml = model_m.to_yaml()
with open('E:\SSEs\pack_output\data\Train_model.yaml', "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model_m.save_weights('E:\SSEs\pack_output\data\Train_weights.h5')
print("Saved model to disk")

model_m.save('E:\SSEs\pack_output\data\model_out.h5')
train_pred = model_m.predict(train_X)

Output = h5py.File("E:\SSEs\pack_output\data\\train_test_val.h5", "w")
with Output as Out_data:
    Out_data['train_X'] = np.squeeze(train_X)
    Out_data['train_Y'] = np.squeeze(train_Y)
    Out_data['test_X'] = np.squeeze(test_X)
    Out_data['test_Y'] = np.squeeze(test_Y)
    Out_data['val_X'] = np.squeeze(val_X)
    Out_data['val_Y'] = np.squeeze(val_Y)

print('ok')
print('ookk')


# test_pred = model_m.predict(test_X)
# val_pred = model_m.predict(val_X)
# Output = h5py.File("E:\SSEs\pack_output\data\D01_60_Nosign_NoUniVal_02\\train_test_val.h5", "w")
# with Output as Out_data:
#     Out_data['train_X'] = np.squeeze(train_X)
#     Out_data['train_Y'] = np.squeeze(train_Y)
#     Out_data['test_X'] = np.squeeze(test_X)
#     Out_data['test_Y'] = np.squeeze(test_Y)
#     Out_data['val_X'] = np.squeeze(val_X)
#     Out_data['val_Y'] = np.squeeze(val_Y)


model_m = tf.keras.models.load_model('E:\SSEs\pack_output\data\\60_day_nosign_8_2_1\model_out.h5')
with h5py.File('E:\SSEs\pack_output\data\\train_test_val.h5','r') as Alldata:
    train_X = Alldata['train_X'][:]
    train_Y = Alldata['train_Y'][:]
    test_X = Alldata['test_X'][:]
    test_Y = Alldata['test_Y'][:]
    val_X = Alldata['val_X'][:]
    val_Y = Alldata['val_Y'][:]

pred_train = model_m.predict(train_X.reshape(-1,60,1))
pred_test = model_m.predict(test_X.reshape(-1,60,1))
pred_val = model_m.predict(val_X.reshape(-1,60,1))

print('train_pred:',(pred_train-train_Y).std()*300,'   train_X:',(train_X-train_Y).std()*300)
print('test_pred:',(pred_test-test_Y).std()*300,'   test_X:',(test_X-test_Y).std()*300)
print('val_pred:',(pred_val-val_Y).std()*300,'   val_X:',(val_X-val_Y).std()*300)

print('okkkk')
print('okkkkkk')



# model_m = tf.keras.models.load_model('E:\SSEs\pack_output\data\\60_day_nosign_8_2_1\model_out.h5')
with h5py.File('E:\\URL_Builder\\outForTrain\\test_real00.h5','r') as Alldata:
    real_X = Alldata['X'][:]
    real_Y = Alldata['Y'][:]
real_X = real_X.reshape(-1,60,1)
real_X_ = np.zeros((5,60,1))
real_X_[0,:,:] = real_X[0,:,:]
real_X_[1,:,:] = real_X[2,:,:]
real_X_[2,:,:] = real_X[4,:,:]
real_X_[3,:,:] = real_X[6,:,:]
real_X_[4,:,:] = real_X[8,:,:]
pred_real = model_m.predict(real_X_)

real_Y_ = np.zeros((5,60))
real_Y_[0,:] = real_Y[0,:]
real_Y_[1,:] = real_Y[2,:]
real_Y_[2,:] = real_Y[4,:]
real_Y_[3,:] = real_Y[6,:]
real_Y_[4,:] = real_Y[8,:]


print('real_pred:',(pred_real-real_Y_).std(),'   real_X:',(np.squeeze(real_X_)-real_Y_).std())

