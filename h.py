

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image



img_width,img_height=800,800
train_data_dir='Dataset/train'
test_data_dir='Dataset/test'
nb_train_samples=1000
nb_test_samples=100
epochs=50
batch_size=20


if k.image_data_format()=='channels_first':
    input_shape=(3, img_width, img_height)
else:
     image_shape=(img_width, img_height, 3)

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1. /255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator=test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#######Create Model
model=Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    test_data=test_generator,
    test_steps=nb_test_samples // batch_size)

model.save_weights('first_try.h5')

img_pred=image.load_img('Dataset/test/1051.png', target_size=(800, 800))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred, axis=0)

result=model.predict(img_pred)
print(result)
if result[0][0]==1:
    prediction="category 1"
elif result[0][0]==2:
    prediction = "category 2"
elif result[0][0]==3:
    prediction = "category 3"
else:
    prediction = "category 4"

print(prediction)








































