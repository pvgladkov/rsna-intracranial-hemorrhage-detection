import numpy as np

import cv2
import keras
from keras_applications.inception_v3 import InceptionV3
import pandas as pd
from rsna.model import Model
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

test_images_dir = '/data/rsna/stage_1_test_images/'
train_images_dir = '/data/rsna/stage_1_train_images/'


def get_tf_session(per_process_gpu_memory_fraction=0.3, num_cores=-1):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    return tf.Session(config=config)


def set_tf_session(per_process_gpu_memory_fraction=0.3):
    set_session(get_tf_session(per_process_gpu_memory_fraction))


def read_testset(filename="/data/rsna//stage_1_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df


def read_trainset(filename="/data/rsna/stage_1_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468, 312469, 312470, 312471, 312472, 312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]

    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df


if __name__ == '__main__':

    set_tf_session(0.4)

    test_df = read_testset()
    df = read_trainset()

    # train set (00%) and validation set (10%)
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)

    # lets go for the first fold only
    train_idx, valid_idx = next(ss)

    # obtain model
    model = Model(engine=InceptionV3, input_dims=(256, 256, 3), batch_size=32, learning_rate=5e-4,
                  num_epochs=8, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1,
                  train_images_dir=train_images_dir, test_images_dir=test_images_dir)

    # obtain test + validation predictions (history.test_predictions, history.valid_predictions)
    history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)

    test_df.iloc[:, :] = np.average(history.test_predictions, axis=0,
                                    weights=[0, 0, 0, 0, 1, 2, 9, 10])  # let's do a weighted average for epochs (>1)

    test_df = test_df.stack().reset_index()

    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

    test_df.to_csv('submission.csv', index=False)

    model.save('/data/rsna/model.h5')

