
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import *
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

class Computer_Vision: # Here we need to build interface and switch on camera in this class.
    def __init__(self):
        self.open = True
    def mainloop(self):
        self.window.mainloop()
        
    def start_camera(self): # i have bug, when this function works, you can close it just when you close whole program, i could use asyncio function so it works together, and i can close it.
        try:
            self.camera = cv2.VideoCapture(0)
            while self.open:
                self.ret, self.frame = self.camera.read()
                self.frame = np.array(self.frame)
                cv2.imshow('Video', self.frame)
                cv2.waitKey(60)
                return self.frame
        except:
            print("Camera can't be opened.")
    def stop_camera(self):
        try:
            self.open = False
            self.camera.release()
            print("Camera stopped successfully.")
        except:
            print("Camera can't be stopped.")
        
    def predict_bounding_box(self, frame, x_left, x_right, y_left, y_right, color, length):
        self.rectangle(frame, (x_left, y_left), (x_right, y_right), color, length)
        
class Model:
    def __init__(self):
        try:
            address_of_saved_model = "/Users/hliblukianov/Documents/computer_vision_model.keras"
            self.model = tf.keras.models.load_model(address_of_saved_model)
            print("Model is found.")
        except:
            self.model = tf.keras.models.Sequential([
               tf.keras.layers.ResizingLayer(width , height)
               tf.keras.layers.Conv2D(3, 3),
               tf.keras.layers.MaxPooling2D(32, 32),
               tf.keras.layers.Conv2D(3, 3),
               tf.keras.layers.MaxPooling2D(8, 8),
                
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(6, activation = 'relu'),
               tf.keras.layers.BatchNormalization(),
               tf.keras.layers.Dense(12, activation = 'relu'),
               tf.keras.layers.Dense(7, activation = 'relu'),
               
            ]) 
            average_layer = tf.keras.layers.GlobalAveragePooling2D()(self.model)
            class_output_layer = tf.keras.layers.Dense(8, activation='softmax')(average_layer)
            coordinates_output = tf.keras.layers.Dense(5)(average_layer)
            self.CV_model = tf.keras.Model(inputs=self.model.input,
                                          outputs=[class_output_layer, coordinates_output])
            
            print("Building new model from scratch.")
        finally:
            print("Initialization completed succesfully")
    def prepare_data(self): 
        print("Preparing data.")
        self.input_shape = (1280, 721)
        self.data = ImageDataGenerator(1/255)
        self.test = ImageDataGenerator(1/255)
        self.validation = ImageDataGenerator(1/255)
        self.dataset = self.data.flow_from_directory("/Users/hliblukianov/Documents/photos_of_food_annotated",
                                                    target_size = self.input_shape,
                                                    color_mode = 'grayscale',
                                                    batch_size = 1,
                                                    shuffle = True,
                                                    class_mode = "binary")

        self.test_dataset = self.test.flow_from_directory("/Users/hliblukianov/Documents/photos_of_food_annotated_test",
                                                         target_size = self.input_shape,
                                                         color_mode = 'grayscale',
                                                         batch_size = 1,
                                                         shuffle = True,
                                                         class_mode = "binary")
        self.validation_dataset = self.validation.flow_from_directory("/Users/hliblukianov/Documents/photos_of_food_annotated_validation",
                                                                     target_size = self.input_shape,
                                                                     color_mode = 'grayscale',
                                                                     batch_size = 1,
                                                                     shuffle = True,
                                                                     class_mode = "binary")
        
    def train_model(self, train_dataset, test_dataset, validation_dataset): 
        print("Model kernel started")
        self.model.compile(optimizer = "nadam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
        self.model.fit(train_dataset, validation_data = test_dataset, epochs=14, batch_size=12)
        self.model.evaluate(validation_dataset)
        self.model.summary()
        self.model.save('/Users/hliblukianov/Documents/computer_vision_model.keras')
        
    def predict_photo(self, frame):
        
        prediction = self.model.predict(frame)
        print(prediction)

model = Model()

model.prepare_data()

model.train_model(model.dataset, model.test_dataset, model.validation_dataset)

cv = Computer_Vision()
print("Camera opened succesfully.")
while True:
    photo = cv.start_camera()
    photo = img_to_array(photo)
    photo = preprocess_input(photo)
    photo = photo.reshape((1, photo.shape[0], photo.shape[1], photo.shape[2]))
    photo = tf.image.rgb_to_grayscale(photo)
    model.predict_photo(photo)

print(model.model.predict(model.dataset[0][0]))
print(model.model.predict(model.dataset[1][0]))
print(model.model.predict(model.dataset[2][0]))
print(model.model.predict(model.dataset[3][0]))
print(model.model.predict(model.dataset[4][0]))
print(model.model.predict(model.dataset[5][0]))
print(model.model.predict(model.dataset[6][0]))
print(model.model.predict(model.dataset[7][0]))

cv.stop_camera()
