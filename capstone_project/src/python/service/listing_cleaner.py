# coding: utf-8


import numpy as np
import pandas as pd
import collections
from ast import literal_eval
import itertools
import unicodedata

from datetime import date

import ipdb


import os
print os.getcwd()


def clean_listing(city_name):

    file_path = ''.join(['../../data/insideAirBnB/benchmark_capitals/',
                         city_name, '/listings.csv'])

    df = pd.read_csv(file_path)
    df = df[df.room_type == 'Entire home/apt']
    df = df[df.availability_90 > 0]


# ## Remove columns without interest

    cols_text = ['name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview',
                 'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about']

    df = df.drop(cols_text, axis=1)

    columns_to_exclude = ['calendar_last_scraped',  'scrape_id', 'listing_url', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 'host_location', 'host_thumbnail_url', 'host_picture_url', 'host_total_listings_count', 'host_acceptance_rate', 'street', 'city', 'state', 'market', 'smart_location', 'country_code', 'country', 'has_availability', 'zipcode', 'neighbourhood', 'host_neighbourhood', 'neighbourhood_cleansed', 'requires_license', 'license', 'jurisdiction_names', 'calculated_host_listings_count', 'square_feet', 'monthly_price',
                          'weekly_price', 'security_deposit', 'cleaning_fee']

    df = df.drop(columns_to_exclude, axis=1)


# ### Remove special characters on columns name function
    def remove_special_character(df, df_name):
        df.columns = [''.join(e for e in col if e.isalnum())
                      for col in df.columns.tolist()]
        df.columns = [df_name + '_' + col for col in df.columns]
        return df


# ## Split the amenities column into single column
    def convert_to_dict(x):
        x = x.replace('{', '').replace('}', '')
        x = x.replace('-', '_')
        x = x.replace('\"', '').replace('\"', '')
        x = x.replace('(', '').replace(')', '')
        x = x.replace('/', '').replace(':', '_')
        x = x.replace('\'', '').replace('.', '_')

        x = x.split(',')
        return collections.Counter(x)

    df.amenities = df.amenities.apply(convert_to_dict)
    df_amenities = df.amenities.apply(pd.Series)
    df_amenities = remove_special_character(df_amenities, 'amenities_')
    df_amenities = df_amenities.fillna(0)

    df = pd.concat([df, df_amenities], axis=1)
    df = df.drop('amenities', axis=1)

    host_verifications2 = df.host_verifications.apply(
        lambda x: collections.Counter(literal_eval(x)))
    df_host_verifications = host_verifications2.apply(pd.Series)
    df_host_verifications.columns = [
        'host_verification_' + col for col in df_host_verifications.columns.tolist()]
    df_host_verifications = df_host_verifications.fillna(0)
    df = pd.concat([df, df_host_verifications], axis=1)
    df = df.drop('host_verifications', axis=1)

    df_bed_type = pd.get_dummies(df.bed_type)
    df_bed_type = remove_special_character(df_bed_type, 'bed_type')
    df = pd.concat([df, df_bed_type], axis=1)
    df = df.drop('bed_type', axis=1)
    print df.shape

    df_neighbourhood_group_cleansed = pd.get_dummies(
        df.neighbourhood_group_cleansed)
    df_neighbourhood_group_cleansed = remove_special_character(
        df_neighbourhood_group_cleansed, 'neighbourhood_group_cleansed_')

    df = pd.concat([df, df_neighbourhood_group_cleansed], axis=1)
    df = df.drop('neighbourhood_group_cleansed', axis=1)

    # ### Convert property_type to categories

    df_property_type = pd.get_dummies(df.property_type)
    df_property_type = remove_special_character(
        df_property_type, 'property_type_')

    df = pd.concat([df, df_property_type], axis=1)
    df = df.drop('property_type', axis=1)

    df.cancellation_policy = df.cancellation_policy.replace('flexible', 0).replace(
        'moderate', 1).replace('strict', 2).replace('super_strict_60', 3)

    # ### Convert last update to days
    def convert_last_update_to_days(x):
        if 'today' in x:
            return 0
        elif 'yesterday' in x:
            return 1
        elif 'a week ago' in x:
            return 7
        elif 'week' in x:
            return int(x.split(' ')[0]) * 7
        elif 'month' in x:
            return int(x.split(' ')[0]) * 30

    df['last_update_days'] = df.calendar_updated.apply(
        convert_last_update_to_days)

    df = df.drop('calendar_updated', axis=1)

    # Convert response time to days

    df.host_response_time = df.host_response_time.replace(
        {'within a few hours': 4, 'within a day': 24, 'within an hour': 1, 'a few days or more': 48})


# ### Convert response rate to integer

    df.host_response_rate = df.host_response_rate.apply(
        lambda x: float(str(x).strip('%')))


# ## Convert price to float

    df.price = df.price.replace('[\$,)]', '', regex=True).replace(
        '[(]', '-', regex=True).astype(float)

    df.extra_people = df.extra_people.replace('[\$,)]', '', regex=True).replace(
        '[(]', '-', regex=True).astype(float)


# ## Convert boolean to integer

    df['host_has_profile_pic'] = (df.host_has_profile_pic == 't') * 1
    df['host_identity_verified'] = (
        df.host_identity_verified == 't') * 1
    df['is_location_exact'] = (df.is_location_exact == 't') * 1
    df['instant_bookable'] = (df.instant_bookable == 't') * 1
    df['require_guest_profile_picture'] = (
        df.require_guest_profile_picture == 't') * 1
    df['require_guest_phone_verification'] = (
        df.require_guest_phone_verification == 't') * 1
    df['host_is_superhost'] = (df.host_is_superhost == 't') * 1


# ## Convert last review and first review into numbers of day

    print df.last_scraped.max()
    last_scraped = pd.to_datetime(df.last_scraped).max().date()

    df.last_review = (last_scraped - pd.to_datetime(df.last_review)).dt.days
    df.first_review = (last_scraped - pd.to_datetime(df.first_review)).dt.days
    df.host_since = (last_scraped - pd.to_datetime(df.host_since)).dt.days

    df.rename(columns={'id': 'listing_id'}, inplace=True)

    df.to_csv(''.join(['../../data/insideAirBnB/benchmark_capitals/',
                       city_name, '/listings_cleansed.csv']), index=False)

    print df.shape, city_name
