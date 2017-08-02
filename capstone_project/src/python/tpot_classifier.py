import numpy as np
import pandas as pd
from tpot import TPOTClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import pickle


df = pd.read_csv('data/all_cities/dataset_for_modeling.csv')

#df = df.sample(frac=.1)

FEATURES = pickle.load(
    open('src/python/best_features/best_features.p', 'rb'))


import ipdb


df_missing_data = pd.DataFrame(
    df.isnull().sum() / len(df) * 100).sort_values(by=0, ascending=False)
df_missing_data.columns = ['missing_percent']
df_missing_data = df_missing_data[df_missing_data.missing_percent > 0]
print df_missing_data


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


X_train, X_test, y_train, y_test, features = create_X_y(FEATURES)

print "X_train.shape : ", X_train.shape


tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')


prediction = tpot.predict(X_test.values)
print confusion_matrix(y_test.values, prediction > .5)
print classification_report(y_test.values, prediction > .5)
