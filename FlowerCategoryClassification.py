#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:48:53 2021

@author: poulomi
"""
'''
This project deals with classifying 3 categories of flower images using deep learning. The flower images is taken 
from the 17FlowerCategoryDataset that originally has flowers of 17 categories and 80 images for each category. After creating
the model, a web application for the same is created using Flask for predicting the category of a new flower image. Finally,
everything is deployed using the Heroku platform.
'''

# importing the necessary libraries

import numpy as np
from glob import glob
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# resizing the images
image_size = [224,224]
training_data = 'train'
valid_data = 'test'

# initializing the resnet50 library and adding the preprocessing layer, here we are using the default imagenet weights
# [224, 224] + [3] making the image RGB channel
# in resnet the output has 1000 categories, but for us it is only 3 categories, so we are not including the first and last
# layer, in the top layer we provide our own data set, thus include_top = False
resnet = ResNet50(input_shape = image_size+[3], weights = 'imagenet', include_top=False)
resnet.summary()

# we do not retrain on existing weights, just retrain on last layer
for layer in resnet.layers:
    layer.trainable = False
    
# fetching the number of output classes, which is 3 for our case
my_folders = glob('train/*')
my_folders

# flatten our resnet output after downloading it
X = Flatten()(resnet.output)

# using the Dense layer to set the length of my folders
prediction = Dense(len(my_folders), activation='softmax')(X)

# creating our model object
my_model = Model(inputs=resnet.input, outputs=prediction)
my_model.summary()

# model compile and optimization
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# reading all the images from the folders, rescaling them and apply data augmentation using ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# only rescaling on test data
test_gen = ImageDataGenerator(rescale=1./255)

# We must provide our image size as target size and not the default (256,256), we keep the batch size and class mode to 
# their default values
train_set = train_gen.flow_from_directory('train', target_size=(224,224), batch_size=32, class_mode='categorical')

test_set = train_gen.flow_from_directory('test', target_size=(224,224), batch_size=32, class_mode='categorical')

# fitting the model
model_r = my_model.fit_generator(train_set, validation_data=test_set, epochs=50, steps_per_epoch=len(train_set), validation_steps=len(test_set))
model_r.history


# plotting the loss
plt.plot(model_r.history['loss'], label='training loss')
plt.plot(model_r.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()

# plotting the accuracy
plt.plot(model_r.history['accuracy'], label='training accuracy')
plt.plot(model_r.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()

# save the model as a h5 file 
my_model.save('model_resnet50.h5')


# Prediction for test data, here the indices mean 0: colt'sfoot, 1:daisy, 2:sunflower
test_pred = my_model.predict(test_set)
test_pred

# choosing the maximum value in a record to determine the class
test_pred = np.argmax(test_pred, axis=1)
test_pred

my_model = load_model('model_resnet50.h5')

################## Also done using a Flask app ##################################
# testing the prediction of the model on a new image 
img = image.load_img('test/Daisy/image_0876.jpg', target_size=(224,224))

# converting to array
x = image.img_to_array(img)
x
x.shape

# rescaling
x = x/225
x

x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
img_data.shape

my_model.predict(img_data)

a = np.argmax(my_model.predict(img_data), axis=1)
a
















