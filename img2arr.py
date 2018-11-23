from PIL import Image
import numpy as np
import csv

# letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"
# def toonehot(text):
#     labellist = []
#     for letter in text:
#         onehot = [0 for _ in range(63)]
#         num = letters.find(letter)
#         onehot[num] = 1
#         labellist.append(onehot)
#     return labellist
# digit = []
#
# for i in range(1,5,1):
#     for j in range(1,11):
#         file = open('./label_csv/train_label'+str(j)+'.csv', 'r')  # csv_file_name
#         reader = csv.reader(file)
#         for row in reader:
#             digit.append(row[i])
#
# digit = np.array(toonehot(digit))
# train_label = digit.reshape(4, -1, 63)
#
#
#
# print(digit)
#
# print(train_label)
# print(train_label.shape)
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
traincsv = open('./test_label2.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/test_captcha2/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])

#==========================rgb2gray
train_data=rgb2gray(train_data)

print(train_data)

# np.save("test_data2.npy",train_data)
# train_data= train_data.reshape(-1,60,160,1)
# print(train_data.shape)