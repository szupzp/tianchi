from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K

working_path = "/home/pzp/PycharmProjects/tianchi/data/tianchi_trainset/"
#K.set_image_dim_ordering('tf')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 0.00001


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1000*(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))

    x = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(x)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2= BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='he_uniform')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_uniform')(conv6)
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_and_predict(use_existing=False):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load(working_path + "trainimages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainmasks.npy").astype(np.float32)

    imgs_val = np.load('/home/pzp/PycharmProjects/tianchi/data/tianchi_valset/' + "valimages.npy").astype(np.float32)
    imgs_mask_val = np.load('/home/pzp/PycharmProjects/tianchi/data/tianchi_valset/' + "valmasks.npy").astype(np.float32)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    mean = np.mean(imgs_val)  # mean for data centering
    std = np.std(imgs_val)  # std for data normalization

    imgs_val -= mean  # images should already be standardized, but just in case
    imgs_val /= std

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('./model/unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir='./logs/Unet_1/', write_images=True, histogram_freq=0)
    #
    # Should we load existing weights?
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')

    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reseasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=8, nb_epoch=20, verbose=1, shuffle=True,validation_data=(imgs_val,imgs_mask_val),
              callbacks=[model_checkpoint,tb])

    # loading best weights from training session
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('./model/unet.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    num_test = len(imgs_val)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_val[i:i + 1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_val[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ", mean)


if __name__ == '__main__':
    train_and_predict(False)