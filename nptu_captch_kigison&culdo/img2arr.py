from PIL import Image
import numpy as np
import csv
import cv2
np.set_printoptions(threshold=np.inf)

# def img2bin(img):
#     ret,thresh = cv2.threshold(img,220,255,cv2.THRESH_BINARY)#二值化
#     return thresh

def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

letters = "0123456789"
def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(10)]
        num = letters.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist

#==========================create_train_data_from_self
traincsv = open('C:/Users/user/Desktop/nptu_captch/nptu_captch.csv', 'r', encoding = 'utf8')
# train_data = np.stack([np.array(Image.open("C:/Users/user/Desktop/captcha/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])
# train_data = np.stack([np.array(img2bin(cv2.imread("C:/Users/user/Desktop/captcha/" + row[0] + ".jpg",cv2.IMREAD_GRAYSCALE)))/255 for row in csv.reader(traincsv)])
train_data = np.stack([np.array(cv2.imread("C:/Users/user/Desktop/nptu_captch/" + row[0] + ".jpg",cv2.IMREAD_GRAYSCALE))/255 for row in csv.reader(traincsv)])
#==========================spider_data
# train_data = np.stack([np.array(Image.open("/home/cbc106013/deep_learning/captcha/nptu3/"+ str(i) + ".jpg"))/255.0 for i in range(0,10000)])
# print(train_data.shape)
# #==========================rgb2gray
# train_data=rgb2gray(train_data)

# print(train_data)

np.save("C:/Users/user/Desktop/nptu_captch/nptu_captch.npy",train_data)
# train_data= train_data.reshape(-1,60,160,1)
print(train_data.shape)