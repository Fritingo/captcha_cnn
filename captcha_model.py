from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv
import cv2

# letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"
letters = "0123456789"

def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(letters.len)]
        num = letters.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist

#==========================train_data
def load_img(path,csv):
    data_csv = open('./'+csv, 'r', encoding = 'utf8')#'test.csv'
    img_data = np.stack([np.array(Image.open(path + row[0] + ".jpg"))/255.0 for row in csv.reader(data_csv)])#/home/cbc106013/deep_learning/captcha/test/
    data_csv.close()
 #==========================rgb2gray
    img_data=rgb2gray(img_data)
    img_data= img_data.reshape(-1,60,160,1)
    print(img_data.shape)
    return img_data
#==========================csv load
# traincsv = open('/home/cbc106013/deep_learning/captcha_cnn/test.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[0]) for row in csv.reader(traincsv)]
# train_label = [[] for _ in range(4)]
# for arr in read_label:
#     for index in range(4):
#         train_label[index].append(arr[index])
# train_label = [arr for arr in np.asarray(train_label)]
def load_label(csv):
    digit = []

    for i in range(1,5,1):
        file = open('./'+csv, 'r', encoding = 'utf8')  # csv_file_name  'test.csv'
        reader = csv.reader(file)
        for row in reader:
            digit.append(row[i])
    file.close()
    print(digit)
    #==========================csv2one_hot
    digit = np.array(toonehot(digit))
    #==========================2y
    data_label = digit.reshape(4,-1,10)
    print(data_label)
    print(data_label.shape)

    data_label = [arr for arr in np.asarray(train_label)]
    print(data_label)
    return data_label
# #==========================test_data
# testcsv = open('./test_y.csv', 'r', encoding = 'utf8')
# test_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/test_y/" + row[0] + ".jpg"))/255.0 for row in csv.reader(testcsv)])

# #==========================rgb2gray
# test_data=rgb2gray(test_data)
# test_data= test_data.reshape(-1,60,160,1)

# #==========================csv load
# # testcsv = open('test_y.csv', 'r', encoding = 'utf8')
# # read_label = [toonehot(row[0]) for row in csv.reader(testcsv)]
# # test_label = [[] for _ in range(4)]
# # for arr in read_label:
# #     for index in range(4):
# #         test_label[index].append(arr[index])
# # test_label = [arr for arr in np.asarray(test_label)]
# # print(test_label)
# digit = []

# for i in range(1,5,1):
#     file = open('test_y.csv', 'r')  # csv_file_name
#     reader = csv.reader(file)
#     for row in reader:
#         digit.append(row[i])

# print(digit)
# #==========================csv2one_hot
# digit = np.array(toonehot(digit))
# #==========================2y
# test_label = digit.reshape(4,-1,10)
# print(test_label)
# print(test_label.shape)
# test_label = [arr for arr in np.asarray(test_label)]

def create_model(train_data, train_label,test_data, test_label):
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
    out = Dropout(0.3)(out)
    out = Flatten()(out)
    out = [Dense(10, name='digit1', activation='softmax')(out),\
        Dense(10, name='digit2', activation='softmax')(out),\
        Dense(10, name='digit3', activation='softmax')(out),\
        Dense(10, name='digit4', activation='softmax')(out)]
    model = Model(input, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()#show

    #==========================儲存最佳辨識率的模型
    filepath="./model/captcha4.h5"
    #==========================每次epoch完會檢查一次，如果比先前最佳的acc高，就會儲存model到filepath
    checkpoint = ModelCheckpoint(filepath, monitor='val_digit3_acc', verbose=1, save_best_only=True, mode='max')
    #===========================也就是在驗證集的val_digit4_acc連續5次不再下降時，就會提早結束訓練
    earlystop = EarlyStopping(monitor='val_digit4_acc', patience=15, verbose=1, mode='auto')
    #===========================圖形化界面
    tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
    #===========================回調函數
    callbacks_list = [checkpoint, earlystop, tensorBoard]
    #===========================
    model.fit(train_data, train_label, batch_size=400, epochs=100, verbose=1, validation_data=(test_data, test_label), callbacks=callbacks_list)
   
if __name__ == "__main__":
    train_data = load_img("/home/cbc106013/deep_learning/captcha/test/","test.csv")
    train_label = load_label("test.csv")
    test_data = load_img("/home/cbc106013/deep_learning/captcha/test_y/","test_y.csv")
    test_label = load_label("test_y.csv")
    create_model(train_data, train_label,test_data, test_label)
    
