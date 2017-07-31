
# coding: utf-8


import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from utils import rstr


def clean_reviews_frequency(city_name):

    root_city_path = ''.join(['../../data/insideAirBnB/benchmark_capitals/',
                              city_name, '/'])

    df = pd.read_csv(root_city_path + 'reviews_metadata.csv',
                     parse_dates=['date'])

    df_listing = pd.read_csv(root_city_path + 'listings_cleansed.csv')

    df_listing_cleansed = df_listing_cleansed[
        df_listing_cleansed.availability_90 > 0]
    df_listing_cleansed = df_listing_cleansed[
        df_listing_cleansed.last_review < 60]

    print df.date.max()
    last_scraped = pd.to_datetime(df_listing.last_scraped).max().date()
    last_scraped
    df['from_scraping_date'] = (last_scraped - df.date).dt.days
    bins = [0, 7, 14, 20, 30, 40, 50, 60, 70, 90, 100, 2000]
    group_names = ["reviewed_more_than_" + str(i) + "_ago" for i in bins[:-1]]
    categories = pd.cut(df['from_scraping_date'], bins, labels=group_names)
    df['review_age'] = categories
    df_group_review_age = df.groupby(['listing_id', 'review_age'])[
        'date'].count()
    df_group_review_age = df_group_review_age.unstack('review_age').fillna(0)

    df_group_review_age.to_csv(root_city_path + 'reviews_frequency.csv')
