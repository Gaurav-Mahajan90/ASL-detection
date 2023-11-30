import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
def load_dataset(data_path):
    data = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150)) 
            data.append(img)
            labels.append(label)
    data = np.array(data) / 255.0 
    labels = np.array(labels)
    return data, labels
data_path = "Data"  
data, labels = load_dataset(data_path)
label_mapping = {label: i for i, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])
labels = tf.keras.utils.to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
image_size = 150
img_channel = 3
model = models.Sequential()
model.add(layers.Conv2D(32,3,activation='relu',padding='same',input_shape = (image_size,image_size,img_channel)))
model.add(layers.Conv2D(32,3,activation='relu',padding='same'))
model.add(layers.MaxPooling2D(padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
model.add(layers.MaxPooling2D(padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128,3,activation='relu',padding='same'))
model.add(layers.Conv2D(128,3,activation='relu',padding='same'))
model.add(layers.MaxPooling2D(padding='same'))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(36, activation='softmax'))
model.summary()
early_stoping = EarlyStopping(monitor='val_loss', min_delta=0.001,patience= 5,restore_best_weights= True, verbose = 0)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, factor=0.5 ,verbose = 1)
model.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stoping,reduce_learning_rate],verbose = 1)
model.save("sign_language_model.h5")