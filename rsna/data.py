from math import ceil
import keras
import numpy as np
import cv2
import pydicom
from rsna.helpers import bsb_window


def _read(path, desired_size):
    """Will be used in DataGenerator"""

    dcm = pydicom.dcmread(path)

    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(desired_size)

    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    return img


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, img_dir, labels=None, batch_size=1, img_size=(512, 512, 1), *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):

        if self.labels is not None:  # for training phase we undersample and shuffle
            # keep probability of any=0 and any=1
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))

        if self.labels is not None:  # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)

            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir + ID + ".dcm", self.img_size)
                Y[i,] = self.labels.loc[ID].values

            return X, Y

        else:  # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir + ID + ".dcm", self.img_size)

            return X