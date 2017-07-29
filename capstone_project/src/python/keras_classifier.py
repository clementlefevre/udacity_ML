import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from keras.wrappers.scikit_learn import KerasClassifier


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping


df = pd.read_csv('data/dataset_for_modeling.csv')


params = {
    'epochs': [300],
    'batch_size': [256]
}


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


def create_X_y():
    TARGET_CLASSIFICATION = 'is_multihost'
    y = df[TARGET_CLASSIFICATION]
    X = df.drop(TARGET_CLASSIFICATION, axis=1)
    rebalanced_index = balanced_subsample(y)
    X, y = X.loc[rebalanced_index], y.loc[rebalanced_index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y)
    features = pd.Series(X.columns)

    return X_train, X_test, y_train, y_test, features


def create_model():

    earlystop = EarlyStopping(
        monitor='val_loss', patience=0, verbose=0, mode='auto')
    # model.add(Dense(256, input_dim=X_train.shape[
    #           1], activation='relu', kernel_initializer='uniform'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[
              1], kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # we can think of this chunk as the hidden layer
    model.add(Dense(128, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # we can think of this chunk as the output layer
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'], callbacks=[earlystop])
    return model


def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=params, cv=n_folds,
                      n_jobs=-1, scoring="recall")
    gs.fit(Xtrain, ytrain)

    return gs.best_estimator_


X_train, X_test, y_train, y_test, features = create_X_y()

print X_train.shape

clf = KerasClassifier(build_fn=create_model, verbose=1)

best_clf = cv_optimize(clf, params, X_train.values, y_train.values)

print(best_clf)

prediction = best_clf.predict(X_test.values)
print confusion_matrix(y_test.values, prediction > .5)
print classification_report(y_test.values, prediction > .5)
