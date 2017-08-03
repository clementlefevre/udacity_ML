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


FILE = ['listings']

count_files = 0

for F in FILE:

    for city in city_folder:

        pictures = [f for f in listdir(
            root_path + city + '/pictures') if isfile(join(root_path + city + '/pictures', f))]

        count_files += len(pictures)

print count_files
