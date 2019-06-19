from __future__ import print_function

import keras
from keras.datasets import mnist,cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.layers import *
import matplotlib.pyplot as plt
import cv2

batch_size = 128
num_classes = 10
epochs = 5

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #mnis #t.load_data()
s=10
x_train = x_train.reshape(50000, 32, 32,3 )  #28,28,1)
x_test = x_test.reshape(10000, 32, 32,3 )  #28,28,1)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train1 = x_train #[y_train==s] #7_8_4
x_test1 = x_test #[y_test==s] #7_8_4
y_train1 = y_train #[y_train==s] #7_8_4
y_test1 = y_test #[y_test==s] #7_8_4

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
x_train1_gray =  rgb2gray(x_train1).reshape(len(x_train1),32,32,1)
x_test1_gray =  rgb2gray(x_test1).reshape(len(x_test1),32,32,1)

# convert class vectors to binary class matrices
y_train1 = keras.utils.to_categorical(y_train1, num_classes)
y_test1 = keras.utils.to_categorical(y_test1, num_classes)


def model_cat(input_image=Input(shape=(32, 32, 1))):  #28,28,1))):
    # A common Conv2D model
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)    
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    x = Dropout(0.5)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf')(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_image, outputs=outputs)

model = model_cat(input_image=Input(shape=(32, 32, 1)))  #  28,28,1)))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.load_weights("./category/cifar10_color/cat_color_weights_"+str(10)+".h5")  #s

def encoder_decoder_model(input_image=Input(shape=(32, 32, 1))): #28,28,1))):
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    #z_mean = Dense(latent_dim, name='z_mean')(x)
    # build decoder model
    #x = Dense(7 * 7 * 64, activation='relu')(x)  #(latent_inputs)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)  #1
    return Model(input_image, outputs)

encoder_decoder = encoder_decoder_model(input_image=Input(shape=(32, 32, 1)))  #28,28,1)))
encoder_decoder.compile(loss='binary_crossentropy',optimizer='adam')

encoder_decoders=[]
for s in range(11):
    encoder_decoder.load_weights("./category/cifar10_color/encoder_decoder_color_weights_"+str(s)+".h5")
    encoder_decoders.append(encoder_decoder)

encoder_decoders[0].summary()

fig=plt.figure(figsize=(32, 8))
size=(320,320)
n=20
def plot_show(img,i,s0,s):
    ax=fig.add_subplot(4,n,i+n*s0+1)
    #ax.set_title(str(s),size=40)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(np.clip(img,0,1))  #original gray image show    

for i in range(n):
    s0=int(np.random.randint(10000))
    img0=x_test1_gray[s0]
    cat=model.predict(img0.reshape(1,32, 32,1 ))  #28,28,1)) #categorize
    s=np.argmax(cat)
    #=0
    img1=encoder_decoders[s].predict(img0.reshape(1,32, 32,1 )) #28,28,1))/255 #cat > individual image
    img1=cv2.resize(img1.reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC) 
    img2=encoder_decoders[10].predict(img0.reshape(1,32, 32,1 )) #28,28,1))/255 #all round image
    
    img2=cv2.resize(img2.reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC) 
    orig_img=cv2.resize(x_test1[s0].reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC)
    img0=cv2.resize(img0.reshape(32,32,1), size,interpolation = cv2.INTER_CUBIC)  #original gray image
    
    plt.gray()
    plot_show(img0,i,0,s)
    plt.title(str(s),size=40)
    
    plot_show(img1,i,1,s)
    plot_show(img2,i,2,s)
    plot_show(orig_img,i,3,s)

plt.savefig("./category/cifar10_color/autodetect_cifar20_test_rand_s.jpg")
plt.show()
plt.close()

fig=plt.figure(figsize=(32, 8))
size=(320,320)
n=20
for i in range(n):
    s0=int(np.random.randint(50000))
    img0=x_train1_gray[s0]
    cat=model.predict(img0.reshape(1,32, 32,1 ))  #28,28,1)) #categorize
    s=np.argmax(cat)
    #=0
    img1=encoder_decoders[s].predict(img0.reshape(1,32, 32,1 )) #28,28,1))/255 #cat > individual image
    img1=cv2.resize(img1.reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC) 
    img2=encoder_decoders[10].predict(img0.reshape(1,32, 32,1 )) #28,28,1))/255 #all round image
    
    img2=cv2.resize(img2.reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC) 
    orig_img=cv2.resize(x_train1[s0].reshape(32,32,3), size,interpolation = cv2.INTER_CUBIC)
    img0=cv2.resize(img0.reshape(32,32,1), size,interpolation = cv2.INTER_CUBIC)  #original gray image
    
    plt.gray()
    plot_show(img0,i,0,s)
    plt.title(str(s),size=40)
    plot_show(img1,i,1,s)
    plot_show(img2,i,2,s)
    plot_show(orig_img,i,3,s)

plt.savefig("./category/cifar10_color/autodetect_cifar20_train_rand_s.jpg")

plt.show()
plt.close()
