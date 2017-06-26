import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2


def load_log_data():
    """
    #reading file path and split data to training and validation data
    """
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for line in reader:
            control = np.random.choice(10,1)
            if control[0] <= 1 or np.abs(float(line[3])) > 0.2:
                lines.append(line)

    #split training file to trainging and validation file
    training_data, validation_data = train_test_split(lines, test_size = 0.2)
    return training_data, validation_data

def augmentation_flip(image, angle):
    """
    #function used to augment training set by flipping the original image horizontally and also reverse turning angle

    """
    flipped_image = np.fliplr(image)
    flipped_angle = -angle
    return flipped_image, flipped_angle


def preprocessing_images(image):
    """
    #function usded to preprocessing images
    #1. corp image to emphasize information in images
    #2. converting corped images to 66*200 size (the same as Nvidia's paper)
    #3. converting BGR color space to YUV color space
    """ 
    #corping the top and bottom image
    new_image = image[50:140,:,:]
    #converting to YUV color space as (nivida's paper's structure)
    new_image = cv2.resize(new_image,(200, 66), interpolation = cv2.INTER_AREA)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV) 
    #normalization
    return new_image



def image_generator(samples, flipped_aug = True, left_right_aug = True, left_right_offset = 0.2, batch_size = 32):
    """
    Fuction used to feed model training. Return 'batch_size' original images
    flipped_aug = True, if you want to include horizontally flipped image for data augmentation. when it equals to True, real batch_size * 2
    (sending a batch a time. Instead of storing all images in memory, only put images needed in memory) 
    """
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles=[]
            for batch_sample in batch_samples:
                name = 'data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                # Converting image to YUV color space, resize to 66 * 200 and normalize
                center_image = preprocessing_images(center_image)
                
                images.append(center_image)
                angles.append(center_angle)
                if flipped_aug == True:
                    flipped_image, flipped_angle = augmentation_flip(center_image, center_angle)
                    images.append(flipped_image)
                    angles.append(flipped_angle)
                # variable to turn on extra images for training (left and right images' truning angles are compensated by offset)
                if left_right_aug == True:
                    left_name = 'data/IMG/' + batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(left_name)
                    left_image = preprocessing_images(left_image)
                    left_angle = center_angle + left_right_offset
                    images.append(left_image)
                    angles.append(left_angle)

                    right_name = 'data/IMG/' + batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(right_name)
                    right_image = preprocessing_images(right_image)
                    right_angle = center_angle - left_right_offset
                    images.append(right_image)
                    angles.append(right_angle)
                    
                    if flipped_aug == True:
                        left_flipped_image, left_flipped_angle = augmentation_flip(left_image, left_angle)
                        right_flipped_image, right_flipped_angle = augmentation_flip(right_image, right_angle)
                        images.append(right_flipped_image)
                        angles.append(right_flipped_angle)
                        images.append(right_flipped_image)
                        angles.append(right_flipped_angle)
                    
            X_image = np.array(images)
            y_angle = np.array(angles)
            yield shuffle(X_image, y_angle)


def model_architecture(keep_prob = 0.5):
    """
    Moddel structure is mimicing from NVIDIA model
    Using dropout for after convolutinal layers and l2 regularizer for reducing overfitting
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape = (66, 200, 3)))
    #The network structre is mimicing from Nvidia' paper
    #First with 3 5x5 convolutional layers
        #First 5x5 convlutional layer 24 filters
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    # model.add(Dropout(0.1))
        #Second 5x5 convlutional layer 36 filters
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    # model.add(Dropout(0.2))
    #model.add(Dropout(keep_prob))
        #Third 5x5 convlutional layer 48 filters
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    # model.add(Dropout(0.2))
    #Dropout
    #model.add(Dropout(keep_prob))
    #Then with 2 1-stride 3x3 convlolutional layers
        #First3x3 convolutional layer
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))

    # model.add(Dropout(0.3))
        #First3x3 convolutional layer
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))


    #Dropout
    # model.add(Dropout(0.3))
    #At last with 3 fully connected layers
        #Flatten the previous results
    model.add(Flatten())
    
        #First fully connected layer
    model.add(Dense(100, init='he_normal',W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.4))
        #Second fully connected layer
    model.add(Dense(50, init='he_normal',W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.4))
        #Third fully connected layer
    model.add(Dense(10, init='he_normal',W_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    #Output layer
    model.add(Dense(1,init='he_normal'))
    model.summary()

    return model

def training_model(model, train_generator, validation_generator, samples_per_epoch,\
                   nb_val_samples, learning_rate =0.0001, nb_epoch = 5):
    """
    fuction used to train model
    variables to control learning process are leanring_rate and nb_epoch
    """
    Adm = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    SGD = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0, nesterov=False)
    #keras.optimizers.Adam(lr = learning_rate)
    model.compile(loss = 'mse', optimizer = SGD)
    history_object = model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch,\
        validation_data=validation_generator, nb_val_samples = nb_val_samples, nb_epoch = nb_epoch)

    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def main():
    """

    """
    # Re-train switch
    reTrain = True
    # Load training data 
    training_data, validation_data = load_log_data()

    # Horizontal flip augmentation switch
    flipped_aug = True
    # Left and right camera pictures switch
    left_right_aug = True
    # Left and righ pictures offset value
    left_right_offset = 0.2
    # batch size value
    batch_size = 40

    # training generator
    training_generator = image_generator(training_data, flipped_aug = flipped_aug, left_right_aug = left_right_aug,\
        left_right_offset = left_right_offset , batch_size = batch_size)
    # validation set generator
    validation_generator = image_generator( validation_data, flipped_aug = flipped_aug,\
        left_right_aug = left_right_aug, left_right_offset = left_right_offset,batch_size = batch_size)

    # This value is not used but has to be a value
    keep_prob = 0.3
    # Build model
    model = model_architecture(keep_prob = keep_prob)

    # If load previous model after adding new sample
    if reTrain == True:
        model.load_weights('model.h5')

    # learning rate
    learning_rate = 0.00001
    # sample number multiplier, calculating how many data after being augmented
    sample_mutiplier = max(2 * flipped_aug,1) * max(3 * left_right_aug, 1) * 1
    samples_per_epoch = len(training_data) * sample_mutiplier
    nb_val_samples = len(validation_data) * sample_mutiplier

    training_model(model, learning_rate = learning_rate,\
    train_generator = training_generator, validation_generator = validation_generator, samples_per_epoch = samples_per_epoch, \
        nb_val_samples =nb_val_samples , nb_epoch = 3 )
    model.save('model.h5', overwrite = True)



if __name__ == '__main__':
    main()




