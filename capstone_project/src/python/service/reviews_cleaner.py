# coding: utf-8
from os import listdir
from os.path import isfile, join
import os

import numpy as np
import pandas as pd
import collections
from ast import literal_eval
import itertools
import unicodedata

from datetime import date

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

from utils import detect_lang

import ipdb

import nltk


def detect_language_reviews(df_reviews, root_city_path):
    print "Start detect lang : {0} : {1} reviews".format(city_name, df_reviews.shape[0])

    df_reviews["comments2"] = df_reviews.comments.apply(
        lambda x: str(x).decode('utf-8'))

    df_reviews['language'] = df_reviews.comments2.apply(detect_lang)

    # drop missing reviews
    df_reviews = df_reviews[~df_reviews.comments.isnull()]

    df_reviews[['listing_id', 'id', 'language']].to_csv(
        ''.join([root_city_path, 'reviews_language.csv']))

    print "Finished detect lang"


def clean_listing(city_name):

    root_city_path = ''.join(['../../data/insideAirBnB/benchmark_capitals/',
                              city_name, '/'])

    file_path = ''.join([root_city_path, 'reviews.csv'])

    df_listing = pd.read_csv(
        ''.join([root_city_path, 'listings_cleansed.csv']))

    print "original listing shape {}".format(df_listing.shape)

    df_listing = df_listing[df_listing.room_type == 'Entire home/apt']
    df_listing = df_listing[df_listing.availability_90 > 0]
    df_listing = df_listing[df_listing.last_review < 60]

    df_reviews = pd.read_csv(
        ''.join([root_city_path, 'reviews.csv']))

    print 'original reviews size for {0} : {1}'.format(city_name, df_reviews.shape[0])

    df_reviews = pd.merge(
        df_listing[['listing_id']], df_reviews, on='listing_id', how='left')

    if not os.path.isfile(root_city_path + 'reviews_language.csv'):
        detect_language_reviews(df_reviews, root_city_path)
    else:
        print "Reviews with lang exists ", city_name

    df_reviews = pd.read_csv(
        ''.join([root_city_path, 'reviews.csv']))

    df_reviews.id = pd.to_numeric(df_reviews.id)

    print "original reviews shape {}".format(df_reviews.shape)
    print "original listing shape {}".format(df_listing.shape)

    df_reviews_language = pd.read_csv(
        ''.join([root_city_path, 'reviews_language.csv']))

    df_reviews_language.id = pd.to_numeric(df_reviews_language.id)

    df_reviews = pd.merge(df_reviews, df_reviews_language[
                          ['id', 'language']], on='id')

    print "reviews after merge with language shape {}".format(df_reviews.shape)

    # if os.path.isfile(root_city_path + 'reviews_PCA.csv'):
    #     return

    # Once each reviews has been labelled, we can proceed to a short
    # analysis regarding the relationships between the languages reviews
    # and the appartment rating.

    df_reviews = pd.merge(df_reviews, df_listing[
                          ['listing_id', 'host_name', 'review_scores_rating']], on='listing_id')

    print " reviews after merge with listing shape {}".format(df_reviews.shape)
    # ### Drop cancelled reservations
    df_reviews.comments = df_reviews.comments.apply(lambda x: str(x))
    df_reviews = df_reviews[~df_reviews.comments.str.contains(
        'The host canceled this reservation')]

    print "reviews after dropping canceled bookings shape {}".format(df_reviews.shape)

    print df_reviews.shape

    def concat_comments(x):
        x = x.str.replace(r'[^a-zA-Z\d\s:]', '')
        return "%s" % '- '.join(x)

    df_reviews_eng = df_reviews[df_reviews.language == 'en']

    df_reviews_eng = df_reviews_eng[
        ~df_reviews_eng.review_scores_rating.isnull()]
    df_reviews_eng = df_reviews_eng[
        ~df_reviews_eng.comments.str.contains('The host canceled this reservation')]
    df_reviews_eng = df_reviews_eng[
        ~df_reviews_eng.comments.str.contains('reservation was canceled')]

    df_reviews_grouped = df_reviews_eng.groupby(
        'listing_id').agg({'comments': concat_comments})
    df_reviews_grouped = df_reviews_grouped.reset_index()
    df_listing_with_reviews = pd.merge(
        df_listing, df_reviews_grouped, on='listing_id', how='left')

    df_listing_with_reviews.comments = df_listing_with_reviews.comments.str.lower()
    df_listing_with_reviews.comments = df_listing_with_reviews.comments.fillna(
        "no comment")
    df_listing_with_reviews.comments = df_listing_with_reviews.comments.astype(
        str)

    print df_listing_with_reviews.shape

    # ### Remove stop words, host names and Stem text reviews.

    # #### Host names

    all_host_names = df_listing.host_name.str.lower()
    all_host_names = all_host_names.unique().tolist()

    # #### Stop words and stem

    porter = nltk.stem.porter.PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    stop = nltk.corpus.stopwords.words('english')
    stop = list(
        set(stop) - set(['no', 'not', 'never', 'don\'t', 'couldn\'t'])) + all_host_names + ['-']

    df_listing_with_reviews.comments = df_listing_with_reviews.comments.apply(
        lambda x: x.replace('no comment', ' '))

    X = df_listing_with_reviews.comments.values

    vectorizer = TfidfVectorizer(ngram_range=(
        2, 3), tokenizer=tokenizer_porter, stop_words=stop, max_features=5000)

    scaler = MinMaxScaler()
    n_components = 30
    svd = TruncatedSVD(n_components=n_components)
    X_all_reviews = vectorizer.fit_transform(X).todense()
    X_all_reviews_TFIDF_scaled = scaler.fit_transform(X_all_reviews)

    X_all_reviews_PCA = svd.fit_transform(X_all_reviews_TFIDF_scaled)
    print svd.explained_variance_ratio_.sum() * 100

    df_reviews_PCA = pd.DataFrame(X_all_reviews_PCA)
    df_reviews_PCA.columns = ['reviews_PC_' +
                              str(i) for i in range(1, n_components + 1)]

    df_listing_with_reviews = pd.concat(
        [df_listing_with_reviews[['listing_id']], df_reviews_PCA], axis=1)

    df_listing_with_reviews.to_csv(
        ''.join([root_city_path, 'reviews_PCA.csv']), index=False)
    print "Finished"
    df_listing_with_reviews.shape
