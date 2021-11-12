import os
import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import model_selection, utils
from random import shuffle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Cropping2D, Lambda, Dense, Dropout
from keras.optimizers import Adam


# Parameters
correction_factor = 0.2
batch_size = 32
epochs = 3

# Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = '/opt/carnd_p3/data/' + batch_sample[0]
                left_name   = '/opt/carnd_p3/data/' + batch_sample[1].split(' ')[-1]
                right_name  = '/opt/carnd_p3/data/' + batch_sample[2].split(' ')[-1]

                # Center camera image
                center_image = mpimg.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Center Camera image flip
                center_image_flip = cv2.flip(center_image, 1)
                center_angle_flip = center_angle * (-1.0)
                images.append(center_image_flip)
                angles.append(center_angle_flip)

                # Left camera image
                left_image = mpimg.imread(left_name)
                left_angle = float(batch_sample[3]) + correction_factor
                images.append(left_image)
                angles.append(left_angle)

                # Left camera image flip
                left_image_flip = cv2.flip(left_image, 1)
                left_angle_flip = left_angle * (-1.0)
                images.append(left_image_flip)
                angles.append(left_angle_flip)

                # Right camera image
                right_image = mpimg.imread(right_name)
                right_angle = float(batch_sample[3]) - correction_factor
                images.append(right_image)
                angles.append(right_angle)

                # Right camera image flip
                right_image_flip = cv2.flip(right_image, 1)
                right_angle_flip = right_angle * (-1.0)
                images.append(right_image_flip)
                angles.append(right_angle_flip)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield utils.shuffle(X_train, y_train)


rows = []
with open('/opt/carnd_p3/data/driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows.append(row)

train_rows, validation_rows = model_selection.train_test_split(rows, test_size=0.2)

train_generator = generator(train_rows, batch_size=batch_size)
validation_generator = generator(validation_rows, batch_size=batch_size)

# Model
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

# Cropping
model.add(Cropping2D(cropping=((70,25), (0,0))))

# 1st conv
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
# model.add(Dropout(0.1))

# 2nd conv
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
# model.add(Dropout(0.1))

# 3rd conv
model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu'))
# model.add(Dropout(0.1))

# 4th conv
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(0.1))

# 5th conv
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))

#Flatten
model.add(Flatten())

# Fully connected
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

# model.summary()


model.compile(loss='mse', optimizer='adam')


history_object = model.fit_generator(train_generator, 
                                    steps_per_epoch=math.ceil(len(train_rows)/batch_size),
                                    validation_data=validation_generator,
                                    validation_steps=math.ceil(len(validation_rows)/batch_size),
                                    epochs=epochs, verbose=1)

model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
