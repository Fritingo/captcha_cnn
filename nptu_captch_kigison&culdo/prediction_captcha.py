from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import csv
letters = "0123456789"

# letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"

model = load_model('C:/Users/user/Desktop/nptu_captch/nptu_captch.h5')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


for i in range(1,400,100):  #predict 份數
    prediction_data = np.stack(np.array(Image.open("C:/Users/user/Desktop/nptu_captch1/"+str(i)+".jpg"))/255.0)  #predict img local
    # print(prediction_data.shape)
    # print("1================================")
    prediction_data=rgb2gray(prediction_data) #灰階

    # print(prediction_data)
    # print("2================================")
    prediction_data= prediction_data.reshape(-1,35,95,1)
    # print(prediction_data)
    # print("3================================")
    prediction = np.array(model.predict(prediction_data))

    # print(prediction)
    # print(max(prediction[0,0,:]))
    ans = []
    p_max = max(prediction[0,0,:])
    for j in range(4):
        p_max = max(prediction[j, 0, :])
        for i in range(len(letters)):
            if (prediction[j,0,i] == p_max ):
                ans.append(letters[i])
                break


    print(ans)