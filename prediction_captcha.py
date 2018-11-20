from keras.models import load_model
import numpy as np
from PIL import Image
import csv
letters = "0123456789"

model = load_model('/home/cbc106013/deep_learning/captcha_cnn/model/captcha4.h5')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

prediction_data = np.stack(np.array(Image.open("/home/cbc106013/deep_learning/captcha/test/6807.jpg"))/255.0)
print(prediction_data.shape)
print("1================================")
prediction_data=rgb2gray(prediction_data)
print(prediction_data)
print("2================================")
prediction_data= prediction_data.reshape(-1,60,160,1)
print(prediction_data)
print("3================================")
prediction = np.array(model.predict(prediction_data))

print(prediction)
print(max(prediction[0,0,:]))
ans = []
p_max = max(prediction[0,0,:])
for j in range(4):
    p_max = max(prediction[j, 0, :])
    for i in range(len(letters)):
        if (prediction[j,0,i] == p_max ):
            ans.append(letters[i])
            break


print(ans)