from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv




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
traincsv = open('./train_label.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/train_captcha/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])

#==========================rgb2gray
train_data=rgb2gray(train_data)

train_data= train_data.reshape(-1,60,160,1)
print(train_data.shape)
#==========================csv load
# traincsv = open('/home/cbc106013/deep_learning/captcha_cnn/test.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[0]) for row in csv.reader(traincsv)]
# train_label = [[] for _ in range(4)]
# for arr in read_label:
#     for index in range(4):
#         train_label[index].append(arr[index])
# train_label = [arr for arr in np.asarray(train_label)]
digit = []

for i in range(1,5,1):
    file = open('train_label.csv', 'r')  # csv_file_name
    reader = csv.reader(file)
    for row in reader:
        digit.append(row[i])

print(digit)
#==========================csv2one_hot
digit = np.array(toonehot(digit))
#==========================2y
train_label = digit.reshape(4,-1,63)
print(train_label)
print(train_label.shape)


train_label = [arr for arr in np.asarray(train_label)]
print(train_label)
#==========================test_data
testcsv = open('./test_label.csv', 'r', encoding = 'utf8')
test_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/test_captcha/" + row[0] + ".jpg"))/255.0 for row in csv.reader(testcsv)])

#==========================rgb2gray
test_data=rgb2gray(test_data)
test_data= test_data.reshape(-1,60,160,1)

#==========================csv load
# testcsv = open('test_y.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[0]) for row in csv.reader(testcsv)]
# test_label = [[] for _ in range(4)]
# for arr in read_label:
#     for index in range(4):
#         test_label[index].append(arr[index])
# test_label = [arr for arr in np.asarray(test_label)]
# print(test_label)
digit = []

for i in range(1,5,1):
    file = open('test_label.csv', 'r')  # csv_file_name
    reader = csv.reader(file)
    for row in reader:
        digit.append(row[i])

print(digit)
#==========================csv2one_hot
digit = np.array(toonehot(digit))
#==========================2y
test_label = digit.reshape(4,-1,63)
print(test_label)
print(test_label.shape)
test_label = [arr for arr in np.asarray(test_label)]

# Create CNN Model
print("Creating CNN model...")
# in = Input((60,160))
input = Input(shape=(60,160,1), name='Input')
out = input
out = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
# out = MaxPooling2D(pool_size=(2, 2))(out)
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
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Flatten()(out)
out = [Dense(63, name='digit1', activation='softmax')(out),\
    Dense(63, name='digit2', activation='softmax')(out),\
    Dense(63, name='digit3', activation='softmax')(out),\
    Dense(63, name='digit4', activation='softmax')(out)]
model = Model(input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()#show

#==========================儲存最佳辨識率的模型
filepath="./model/captcha4.h5"
#==========================每次epoch完會檢查一次，如果比先前最佳的acc高，就會儲存model到filepath
checkpoint = ModelCheckpoint(filepath, monitor='val_digit3_acc', verbose=1, save_best_only=True, mode='max')
#===========================也就是在驗證集的val_digit4_acc 下降後60次提前停止訓練
#earlystop = EarlyStopping(monitor='val_digit3_acc', patience=60, verbose=1, mode='auto')
#===========================圖形化界面
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
#===========================回調函數
callbacks_list = [checkpoint, tensorBoard] #earlystop
#===========================
model.fit(train_data, train_label, batch_size=400, epochs=150, verbose=1, validation_data=(test_data, test_label), callbacks=callbacks_list)
