# from __future__ import print_function
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential  # Sequential模型
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# import matplotlib
# from keras.utils.vis_utils import plot_model
#
# batch_size = 128  # 每梯處理數目
# num_classes = 10  # 類別數
# epochs = 12  # 梯數
#
# # input image dimensions
# img_rows, img_cols = 28, 28  # 橫、縱
#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# if K.image_data_format() == 'channels_first'  # 如果channels(RGB)在前
#     # 轉x_train.shape(60000, 1, 28, 28)___channel=1(灰度圖)
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else
#     # 轉x_train.shape(60000, 28, 28, 1)___channel=1(灰度圖)
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')  # 轉float32
# x_test = x_test.astype('float32')
# x_train = 255  # 轉0~1
# x_test = 255
# print('x_train shape', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)  # 轉成one-hot
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()  # Sequential模型
# # 第一層 二維卷積
# # 32 filters卷積核數目、輸出數
# # kernel_size 卷稽核大小 3X3
# # 激活函數(主要作用是引入非線性) relu(通常用relu 好處避免overfit 不用指數計算量小
# # 第一層必須加input_shape
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# # 加一層 64 filters
# model.add(Conv2D(64, (3, 3), activation='relu'))
# #
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # 加池化 用2X2pooling
# model.add(Dropout(0.25))  # 捨棄神經元25%(避免overfit)
#
# model.add(Flatten())  # 攤平  數據一維化
# # 加Dense 全連接層 128輸出維度
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))  # 捨棄神經元50%
# # 加Dense 全連接層 10(答案)輸出  激活函數用softmax(歸一化指數函數)
# model.add[Dense(34, name='digit1', activation='softmax'),\
#     Dense(34, name='digit2', activation='softmax'),\
#     Dense(34, name='digit3', activation='softmax'),\
#     Dense(34, name='digit4', activation='softmax'),\
#     Dense(34, name='digit5', activation='softmax')]
#
# # loss 損失函數 categorical_crossentropy(多類對數損失，轉換成二值序列，需先one-hot)
# # optimizer 優化器 (默認keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06))
# # lr學習率(=0的浮點數)  rho(=0的浮點數) epsilon防止除0錯誤(=0的浮點數)
# # metrics 性能評估 (自訂義) 準確性
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,  # 輸入數據
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,  # 顯示紀錄 (0不輸出紀錄  1輸出進度條  2每代紀錄)
#           validation_data=(x_test, y_test))  # 形成(x,y)的tuple (指定驗證用)
# # print最後的loss accuracy
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss', score[0])
# print('Test accuracy', score[1])
# # 生成modle圖
# plot_model(model, to_file='model1.png', show_shapes=True)
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv
import cv2

digit1 = []
digit2 = []
digit3 = []
digit4 = []


def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"
# letters = "0123456789"
def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(63)]
        num = letters.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist

#==========================train_data
traincsv = open('./test.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/test/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])

#==========================rgb2gray
train_data=rgb2gray(train_data)

train_data= train_data.reshape(10,60,160,1)
print(train_data.shape)
#==========================csv load
# digit = []
#
# for i in range(1,5,1):
#     file = open('test.csv', 'r')  # csv_file_name
#     reader = csv.reader(file)
#     for row in reader:
#         digit.append(row[i])
#
# print(digit)
# #==========================csv2one_hot
# digit = np.array(toonehot(digit))
# #==========================2y
# train_label = digit.reshape(4,-1,63)
# print(train_label)
# print(train_label.shape)
traincsv = open('test.csv', 'r', encoding = 'utf8') # 讀取訓練集的標記
read_label = [toonehot(row[0]) for row in csv.reader(traincsv)] # 將每一行的文字轉成one-hot encoding
train_label = [[] for _ in range(4)] # 各組輸出的答案要放到train_label

for arr in read_label:
    for index in range(4):
        train_label[index].append(arr[index]) # 原本是[[第1字答案, ..., 第6字答案],......, [第1字答案, ..., 第6字答案]]
                                              # 要轉成[[第1字答案,..., 第1字答案],..., [第6字答案,..., 第6字答案]]才符合Y的輸入
train_label = [arr for arr in np.asarray(train_label)]
#==========================test_data
testcsv = open('./test_y.csv', 'r', encoding = 'utf8')
test_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/test_y/" + row[0] + ".jpg"))/255.0 for row in csv.reader(testcsv)])

#==========================rgb2gray
test_data=rgb2gray(test_data)
test_data= test_data.reshape(10,60,160,1)

#==========================csv load
testcsv = open('test_y.csv', 'r', encoding = 'utf8') # 讀取訓練集的標記
read_label = [toonehot(row[0]) for row in csv.reader(testcsv)] # 將每一行的文字轉成one-hot encoding
test_label = [[] for _ in range(4)] # 各組輸出的答案要放到train_label

for arr in read_label:
    for index in range(4):
        test_label[index].append(arr[index]) # 原本是[[第1字答案, ..., 第6字答案],......, [第1字答案, ..., 第6字答案]]
                                              # 要轉成[[第1字答案,..., 第1字答案],..., [第6字答案,..., 第6字答案]]才符合Y的輸入
test_label = [arr for arr in np.asarray(test_label)]
# digit = []
#
# for i in range(1,5,1):
#     file = open('test_y.csv', 'r')  # csv_file_name
#     reader = csv.reader(file)
#     for row in reader:
#         digit.append(row[i])
#
# print(digit)
# #==========================csv2one_hot
# digit = np.array(toonehot(digit))
# #==========================2y
# test_label = digit.reshape(4,-1,63)
# print(test_label)
# print(test_label.shape)
# Create CNN Model
print("Creating CNN model...")
# in = Input((60,160))
input = Input(shape=(60,160,1), name='Input')
out = input
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.3)(out)
out = [Dense(63, name='digit1', activation='softmax')(out),\
    Dense(63, name='digit2', activation='softmax')(out),\
    Dense(63, name='digit3', activation='softmax')(out),\
    Dense(63, name='digit4', activation='softmax')(out)]
model = Model(input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()#show

# print("Reading training data...")
# traincsv = open('./data/5_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')
# train_data = np.stack([np.array(Image.open("./data/5_imitate_train_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])
# traincsv = open('./data/5_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
# train_label = [[] for _ in range(5)]
# for arr in read_label:
#     for index in range(5):
#         train_label[index].append(arr[index])
# train_label = [arr for arr in np.asarray(train_label)]
# print("Shape of train data:", train_data.shape)
#
# print("Reading validation data...")
# valicsv = open('./data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
# vali_data = np.stack([np.array(Image.open("./data/5_imitate_vali_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(valicsv)])
# valicsv = open('./data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
# vali_label = [[] for _ in range(5)]
# for arr in read_label:
#     for index in range(5):
#         vali_label[index].append(arr[index])
# vali_label = [arr for arr in np.asarray(vali_label)]
# print("Shape of validation data:", vali_data.shape)

#==========================儲存最佳辨識率的模型
filepath="./model/captcha4.h5"
#==========================每次epoch完會檢查一次，如果比先前最佳的acc高，就會儲存model到filepath
checkpoint = ModelCheckpoint(filepath, monitor='val_digit4_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_digit4_acc', patience=100, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=400, epochs=100, verbose=2, validation_data=(test_data, test_label), callbacks=callbacks_list)