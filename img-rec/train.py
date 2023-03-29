# Derek Brown - student ID:005836312 - capstone project
# Some sections of code in this file were adapted from the following source:
# Menon K., How To Build Powerful Keras Image Classification Models: Simplilearn (2022),
# https://www.simplilearn.com/tutorials/deep-learning-tutorial/guide-to-building-powerful-keras-image-classification-models
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
import cv2
import tensorflow as tf

CLASSES = ['positive', 'negative']
CLASS_LABELS = {class_label: i for i, class_label in enumerate(CLASSES)}

# size for all the images
IMG_SIZE = (128, 128)
DATA_DIR = r"resources\data"
acc_data = None


# This will load the data for training
def data_loader():
    output = []

    for data_folder in os.listdir(DATA_DIR):
        images = []
        labels = []

        d_folder = os.path.join(DATA_DIR, data_folder)
        for subfolder in os.listdir(d_folder):
            cur_folder = os.path.join(d_folder, subfolder)
            cur_label = CLASS_LABELS[subfolder]

            for file in os.listdir(cur_folder):
                img_file = os.path.join(cur_folder, file)

                cur_img = cv2.imread(img_file)
                cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
                cur_img = cv2.resize(cur_img, IMG_SIZE)

                images.append(cur_img)
                labels.append(cur_label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))
        pickle.dump(output, open('resources/loaded_data.pkl', 'wb'))
    return


# Calling this function will both load the data and train the model
def train_model():

    loaded_data = pickle.load(open('resources/loaded_data.pkl', 'rb'))
    global train_images, train_labels, test_images, test_labels
    (test_images, test_labels), (train_images, train_labels) = loaded_data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='selu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    global acc_data
    acc_data = model.fit(train_images, train_labels, batch_size=10, epochs=6, validation_data=(test_images, test_labels))
    # These two lines save the model and accuracy data
    pickle.dump(model, open('resources/trained_model.pkl', 'wb'))
    pickle.dump(acc_data, open('resources/model_accuracy.pkl', 'wb'))
