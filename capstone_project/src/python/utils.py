import math
from langdetect import detect


'''language detection
'''


def detect_lang(comment):

    language = "No found"
    try:
        language = detect(comment)
    except:
        print "could not find language"
    return language


''' R-like str function
'''


def get_unique(x):
    return type(x.values[1]), x.unique()


def rstr(df):
    return df.apply(get_unique)



def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d
