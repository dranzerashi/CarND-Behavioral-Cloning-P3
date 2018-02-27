from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import Cropping2D
import gc;
import csv
import numpy as np
import cv2
from sklearn.linear_model import ElasticNet

image_dir = './data/IMG/'
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[1:]
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

training_samples, validation_samples = train_test_split(samples,test_size=0.2)


def batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset: offset+batch_size]
            # Load images
            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 1
                file_path_center = image_dir + batch_sample[0].split('/')[-1]
                file_path_left = image_dir + batch_sample[1].split('/')[-1]
                file_path_right = image_dir + batch_sample[2].split('/')[-1]
                center_image = cv2.imread(file_path_center)
                center_angle = float(batch_sample[3])
                right_image = cv2.imread(file_path_right)
                right_angle = float(batch_sample[3]) - correction
                left_image = cv2.imread(file_path_left)
                left_angle = float(batch_sample[3]) + correction
                # print(center_image.shape)
                images.append(center_image)
                angles.append(center_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(left_image)
                angles.append(left_angle)

                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                right_image_flipped = np.fliplr(right_image)
                right_angle_flipped = -right_angle
                left_image_flipped = np.fliplr(left_image)
                left_angle_flipped = -left_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
                images.append(right_image_flipped)
                angles.append(right_angle_flipped)
                images.append(left_image_flipped)
                angles.append(left_angle_flipped)


            x_train = np.array(images)
            y_train = np.array(angles)
            gc.collect()
            yield shuffle(x_train , y_train)


train_generator = batch_generator(training_samples, batch_size=64)
validation_generator = batch_generator(validation_samples, batch_size=64)

ch, row, col = 3, 80, 320  # Trimmed image format
# print(next(train_generator)[0].shape)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255.0)-0.5,output_shape=(90, 320, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.summary()
model.fit_generator(train_generator, samples_per_epoch=len(training_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,nb_epoch=10)
model.save('model.h5')
