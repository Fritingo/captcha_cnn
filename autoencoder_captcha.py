from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

input_img = Input(shape=(35, 95, 1),name='Input')    # adapt this if using 'channels_first' image data format
x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((5, 5), padding='same')(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((5, 5), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((5, 5), padding='same')(x)

# at this point the representation is (4, 4, 8), i.e. 128-dimensional

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((5, 5))(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((5, 5))(x)
# x = Conv2D(8, (3, 3), activation='relu')(x)
# x = UpSampling2D((5, 5))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# To train it, use the original MNIST digits with shape (samples, 3, 28, 28),
# and just normalize pixel values between 0 and 1

# (x_train, _), (x_test, _) = mnist.load_data()
#
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))    # adapt this if using 'channels_first' image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))       # adapt this if using 'channels_first' image data format
# print(x_train.shape)
# print(x_test.shape)
for i in range(1,3):
    print(i)
    train_data_temporary = np.load("/home/cbc106013/deep_learning/captcha/img_data_gray/nptu_create"+str(i)+".npy")
    if i == 1:
        train_data = train_data_temporary
    else:
        train_data = np.concatenate((train_data,train_data_temporary),axis=0)
for i in range(1,3):
    print(i)
    train_data_temporary = np.load("/home/cbc106013/deep_learning/captcha/img_data_gray/nptu"+str(i)+".npy")
    train_data = np.concatenate((train_data,train_data_temporary),axis=0)

train_data= train_data.reshape(-1,35,95,1)
print(train_data.shape)
x_train = train_data
print(x_train.shape)

test_data= np.load("/home/cbc106013/deep_learning/captcha/img_data_gray/nptu3.npy")

test_data_temporary = np.load("/home/cbc106013/deep_learning/captcha/img_data_gray/nptu_create3.npy")

test_data = np.concatenate((test_data,test_data_temporary),axis=0)

test_data= test_data.reshape(-1,35,95,1)
print(test_data.shape)
x_test = test_data
print(x_train.shape)
print(x_test.shape)
# open a terminal and start TensorBoard to read logs in the autoencoder subdirectory
# tensorboard --logdir=autoencoder

autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=1)

# take a look at the reconstructed digits
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(10, 4), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(35, 95))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(35, 95))
    plt.gray()
    ax.set_axis_off()

plt.show()

# take a look at the 128-dimensional encoded representation
# these representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images

encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)

# save latent space features 128-d vector
pickle.dump(encoded_imgs, open('conv_autoe_features.pickle', 'wb'))

n = 10
plt.figure(figsize=(10, 4), dpi=100)
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.set_axis_off()

plt.show()

K.clear_session()
encoder.save('./model/autoencoder_captcha.h5')