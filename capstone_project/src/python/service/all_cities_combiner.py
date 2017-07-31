import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import listing_cleaner
import reviews_cleaner
import pictures_cleaner

root_path = '/home/ramon/workspace/udacity_ML/capstone_project/data/insideAirBnB/benchmark_capitals/'
city_folder = [f for f in listdir(root_path)]
city_folder.sort()

print city_folder

df_all_listings_cleansed = pd.DataFrame()


FILE = ['pictures_contrasts_brightness']

for F in FILE:

    for city in city_folder:

        df = pd.read_csv(root_path + city +
                         '/' + F + '.csv', index_col=False)
        df.columns = map(str.lower, df.columns)

        if F == 'reviews_PCA':
            df = df.iloc[:, 0:20]

        df['city'] = city
        df_all_listings_cleansed = pd.concat(
            [df_all_listings_cleansed, df], axis=0, ignore_index=True)

    df_all_listings_cleansed.to_csv(
        root_path + F + '_all.csv', index=False)
