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


for city in city_folder:
    pictures_cleaner.clean_pictures(city)
