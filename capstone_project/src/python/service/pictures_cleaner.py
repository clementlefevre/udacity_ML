
# coding: utf-8

import os
import time
from os import listdir
from os.path import isfile, join


import numpy as np
import pandas as pd


import scipy
import scipy.misc
import scipy.cluster

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


from PIL import Image, ImageFilter
import urllib
import ipdb

# Pictures paramters
SIZE = (64, 64)
FORMAT = 'L'


############################################################################
##### Scrap pictures                                    ####################
############################################################################

def check_pictures_files_is_local(list_url, root_city_path):
    local_files = [f for f in listdir(
        root_city_path + 'pictures') if isfile(join(root_city_path + 'pictures', f))]

    local_files = [int(f.split(".")[0]) for f in local_files]

    server_files = [f for f in list_url if f[0] not in local_files]

    return server_files

    list_url = check_pictures_files_is_local(list_url, root_city_path)


def scrap_pictures(id, url, root_city_path):

    try:
        urllib.urlretrieve(url, root_city_path +
                           "pictures/" + str(id) + ".jpg")
        time.sleep(.1)

    except Exception as e:
        print e
        print "cannot scrap {}".format(url)


############################################################################
##### Compute contrast and brightness for each picture ####################
############################################################################
def compute_contrast_and_brightness(pic_file, root_city_path):
    pic_id = pic_file.split('.')[0]
    img = Image.open(root_city_path + 'pictures/' + pic_file).convert('RGB')
    img = img.resize(SIZE, Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="int32")
    R = data[:, :, 0]
    G = data[:, :, 1]
    B = data[:, :, 2]
    LuminanceA = (0.2126 * R) + (0.7152 * G) + (0.0722 * B)

    return np.asarray((pic_id, R.std(), G.std(), B.std(), LuminanceA.mean(), LuminanceA.std()), dtype="int32")


def create_contrast_brightness_matrix(root_city_path, list_pics):

    i = 0
    for pic_file in list_pics:
        try:
            npdata = compute_contrast_and_brightness(pic_file, root_city_path)
        except Exception as e:
            print "{0} for {1}".format(e, pic_file)
            continue
        if i == 0:
            npdata_all = npdata
            i = 1
        else:
            npdata_all = np.vstack((npdata_all, npdata))
    return npdata_all


def create_contrast_brightness_df(root_city_path, list_pics):

    # if os.path.isfile(root_city_path + 'pictures_contrasts_brightness.csv'):
    #     return

    df_contrasts_brightness = pd.DataFrame(
        create_contrast_brightness_matrix(root_city_path, list_pics))

    df_contrasts_brightness.columns = ['listing_id', 'R_contrast',
                                       'G_contrast', 'B_contrast', 'Brightness_mean', 'Brightness_std']
    df_contrasts_brightness.to_csv(
        root_city_path + 'pictures_contrasts_brightness.csv', index=False)


############################################################################
##### Convert all pictures to greyscale array                         #####
############################################################################
def load_image(infilename):
    img = Image.open(infilename).convert(FORMAT)
    img = img.resize(SIZE, Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="int32")

    # save_image(data)
    data = data.reshape(1, -1)
    return data


def reconstruct_image(npdata):
    clipped_ndata = np.asarray(np.clip(npdata, 0, 255), dtype="uint8")
    img = Image.fromarray(clipped_ndata, FORMAT)
    img.show()


def create_matrix_greyscale_pictures(root_city_path, list_pics):

    i = 0
    for pic_file in list_pics:
        pic_id = pic_file.split('.')[0]
        try:
            npdata = load_image(root_city_path + 'pictures/' + pic_file)
        except Exception as e:
            print "{0} for {1}".format(e, pic_file)
            continue
        npdata = np.append([pic_id], npdata)

        if i == 0:

            npdata_all = npdata
            i = 1
        else:
            npdata_all = np.vstack((npdata_all, npdata))

    return npdata_all


def create_greyscale_df(root_city_path, list_pics):
    if os.path.isfile(root_city_path + 'pictures_greyscale.csv'):
        return
    npdata_all = create_matrix_greyscale_pictures(root_city_path, list_pics)
    df_pictures = pd.DataFrame(npdata_all)
    cols = ["pix_" + str(i) for i in range(0, df_pictures.shape[1] - 1)]
    df_pictures.columns = ['listing_id'] + cols
    df_pictures.to_csv(root_city_path + 'pictures_greyscale.csv', index=False)


############################################################################
##### PCA on greyscale                                                 #####
############################################################################


def create_greyscale_PCA(root_city_path):
    if os.path.isfile(root_city_path + 'pictures_PCA.csv'):
        return

    df_pictures = pd.read_csv(
        root_city_path + 'pictures_greyscale.csv', index_col=False)
    cols = ["pix_" + str(i) for i in range(0, df_pictures.shape[1] - 1)]
    pca = PCA(n_components=50)

    X_train = scale(df_pictures[cols].values)

    X_transformed = pca.fit_transform(X_train)

    df_pictures_PCA = pd.DataFrame(X_transformed)
    df_pictures_PCA.columns = ['pictures_PC_' + str(i) for i in range(1, 51)]

    df_pictures_PCA = pd.concat(
        [df_pictures.listing_id, df_pictures_PCA], axis=1)

    df_pictures_PCA.to_csv(root_city_path + 'pictures_PCA.csv')


############################################################################
##### Top COLORS                                                      #####
############################################################################

def extract_top_colour_in_picture(pic_file, root_city_path):
    NUM_CLUSTERS = 5

    pic_id = np.asarray([pic_file.split('.')[0]])

    im = Image.open(root_city_path + 'pictures/' + pic_file)
    im = im.resize((150, 150))      # optional, to reduce time
    ar = scipy.misc.fromimage(im)
    ar = ar.astype(float)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2])

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    # print 'cluster centres:\n', codes

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]

    peak = peak.astype(int)
    colour = ''.join(chr(c) for c in peak).encode('hex')
    # print 'most frequent is %s (#%s) \n' % (peak, colour)

    i = 0

    for c in range(NUM_CLUSTERS):
        c_centroid = np.asarray(codes[c])
        c_counts = np.asarray([counts[c]])
        c_hex = ''.join(chr(c) for c in c_centroid.astype(int)).encode('hex')
        c_hex = np.asarray([c_hex])

        if i == 0:
            npdata = np.concatenate(
                (pic_id, c_centroid, c_counts, c_hex))  # .astype('int32')
            i = 1
        else:
            # .astype('int32')))
            npdata = np.hstack(
                (npdata, np.concatenate((c_centroid, c_counts, c_hex))))

    return npdata


def create_colour_clusters_matrix(root_city_path, list_pics):

    i = 0
    for pic_file in list_pics:

        try:
            npdata = extract_top_colour_in_picture(pic_file, root_city_path)
        except Exception as e:
            print "{0} for {1}".format(e, pic_file)
            continue

        if i == 0:

            npdata_all = npdata
            i = 1
        else:
            npdata_all = np.vstack((npdata_all, npdata))

    return npdata_all


def create_colour_clusters_df(root_city_path, list_pics):
    if os.path.isfile(root_city_path + 'pictures_colors_clusters.csv'):
        return
    df_colors_cluster = pd.DataFrame(
        create_colour_clusters_matrix(root_city_path, list_pics))
    df_colors_cluster.columns = ['listing_id', 'Centroid_R_1', 'Centroid_G_1', 'Centroid_B_1', 'Centroid_Count_1', 'Centroid_hex_1',
                                 'Centroid_R_2', 'Centroid_G_2', 'Centroid_B_2', 'Centroid_Count_2', 'Centroid_hex_2',
                                 'Centroid_R_3', 'Centroid_G_3', 'Centroid_B_3', 'Centroid_Count_3', 'Centroid_hex_3',
                                 'Centroid_R_4', 'Centroid_G_4', 'Centroid_B_4', 'Centroid_Count_4', 'Centroid_hex_4',
                                 'Centroid_R_5', 'Centroid_G_5', 'Centroid_B_5', 'Centroid_Count_5', 'Centroid_hex_5']
    cols_int = [col for col in df_colors_cluster.columns if 'hex' not in col]
    df_colors_cluster[cols_int] = df_colors_cluster[
        cols_int].apply(pd.to_numeric)
    df_colors_cluster.to_csv(
        root_city_path + 'pictures_colors_clusters.csv', index=False)


############################################################################
##### MAIN METHOD                                                      #####
############################################################################

def clean_pictures(city_name):

    print 'city :', city_name

    root_city_path = ''.join(['../../data/insideAirBnB/benchmark_capitals/',
                              city_name, '/'])

    file_path = ''.join([root_city_path, 'listings.csv'])

    if not os.path.exists(''.join([root_city_path, '/pictures'])):
        os.makedirs(''.join([root_city_path, '/pictures']))

    df_listing = pd.read_csv(file_path)

    df_listing = df_listing[df_listing.room_type == 'Entire home/apt']
    df_listing = df_listing[df_listing.availability_90 > 0]

    list_url = df_listing[['id', 'picture_url']].sort_values(
        by='id').values.tolist()

    list_url = check_pictures_files_is_local(list_url, root_city_path)
    print len(list_url)
    for url in list_url:
        scrap_pictures(url[0], url[1], root_city_path)

    df_listing_cleansed = pd.read_csv(
        ''.join([root_city_path, 'listings_cleansed.csv']))

    print "before filtering on active = ", df_listing_cleansed.shape
    df_listing_cleansed = df_listing_cleansed[
        df_listing_cleansed.availability_90 > 0]
    df_listing_cleansed = df_listing_cleansed[
        df_listing_cleansed.last_review < 60]

    print "after filtering on active = ", df_listing_cleansed.shape

    list_pics = [str(
        listing_id) + '.jpg' for listing_id in df_listing_cleansed.listing_id.tolist()]

    print len(list_pics)

    create_contrast_brightness_df(root_city_path, list_pics)
    # create_greyscale_df(root_city_path, list_pics)
    # create_greyscale_PCA(root_city_path)
    #create_colour_clusters_df(root_city_path, list_pics)
