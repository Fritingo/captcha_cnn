import numpy as np
import cv2
import matplotlib.pyplot
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join

np.set_printoptions(threshold=np.inf)
#2019/01/1 改成膨脹、侵蝕，整體改灰階


#OpenCV定义的结构元素
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

path = 'C:/Users/lab-pc3/Desktop/DenseNet/nptu_captch/'
files = listdir(path)  # 取得所有檔案與子目錄名稱
for f in files:
    img = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)
    ret,thresh = cv2.threshold(img,20,255,cv2.THRESH_TRUNC)#閾值濾波
    thresh = cv2.blur(thresh, (2,1))#高斯模糊
    thresh = cv2.medianBlur(thresh,3)#平滑
    thresh = cv2.blur(thresh, (2,3))#高斯模糊
    # ret,thresh = cv2.threshold(thresh,18,255,cv2.THRESH_BINARY)#二值化
    # thresh = cv2.dilate(thresh,kernel)#膨脹
    # thresh = cv2.erode(thresh,kernel)#侵蝕
    thresh = thresh*12
    # print(thresh)
    # print(img)

    cv2.imwrite(path+f[0:4]+"(trunc)"+".jpg", thresh)

# titles = ["IMG","TRUNC"]
# images = [img,thresh]
#
# for i in range(2):
#     matplotlib.pyplot.subplot(1,2,i+1)
#     matplotlib.pyplot.imshow(images[i],"gray")
#     matplotlib.pyplot.title(titles[i])
#     matplotlib.pyplot.xticks([])
#     matplotlib.pyplot.yticks([])
#
# matplotlib.pyplot.show()