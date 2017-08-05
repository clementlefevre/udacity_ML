import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.optimizers import RMSprop


import pickle


df = pd.read_csv('data/all_cities/dataset_for_modeling.csv')


FEATURES = pickle.load(
    open('src/python/best_features/features_Kbest_fclass.p', 'rb'))

print len(FEATURES)

FEATURES = list(set(df.columns.tolist()).intersection(set(FEATURES)))

print len(FEATURES)
import ipdb


df_missing_data = pd.DataFrame(
    df.isnull().sum() / len(df) * 100).sort_values(by=0, ascending=False)
df_missing_data.columns = ['missing_percent']
df_missing_data = df_missing_data[df_missing_data.missing_percent > 0]
print df_missing_data

#df = df.sample(frac=.3)


filepath = "weights-improvement-best.hdf5"


earlystop = EarlyStopping(
    monitor='val_loss', patience=200,  verbose=1, mode='auto')

checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def balanced_subsample(y, size=None):

    subsample = []

    if size is None:

        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample


def create_X_y(FEATURES):
    TARGET_CLASSIFICATION = 'multihost'
    y = df[TARGET_CLASSIFICATION]
    X = df.drop(TARGET_CLASSIFICATION, axis=1)
    X = df[FEATURES]
    rebalanced_index = balanced_subsample(y)
    X, y = X.loc[rebalanced_index], y.loc[rebalanced_index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2)
    features = pd.Series(X.columns)

    return X_train, X_test, y_train, y_test, features


def create_model():

    model = Sequential()

    ####################################################
    ############### MODEL 1 ############################
    ####################################################

    # model.add(Dense(256, input_dim=X_train.shape[
    #           1], activation='relu', kernel_initializer='uniform'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    ####################################################
    ############### MODEL 2 ############################
    ####################################################

    model.add(Dense(128, input_dim=X_train.shape[
              1], kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the hidden layer
    model.add(Dense(64, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    ####################################################
    ############### COMPILE############################
    ####################################################

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    return model


X_train, X_test, y_train, y_test, features = create_X_y(FEATURES)


print "X_train.shape : ", X_train.shape
print "X_test.shape : ", X_test.shape

model = create_model()
model.fit(X_train.values, y_train.values, epochs=10000,
          batch_size=256, validation_split=0.2, shuffle=True, callbacks=[earlystop, checkpoint])


prediction = model.predict(X_test)
print confusion_matrix(y_test, prediction > .5)
print classification_report(y_test, prediction > .5)
