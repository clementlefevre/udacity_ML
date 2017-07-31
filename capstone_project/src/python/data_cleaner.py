
# coding: utf-8

import numpy as np
import pandas as pd
import collections
import itertools
import unicodedata

from datetime import date
import seaborn as sns
from ast import literal_eval

from utils import rstr, detect_lang


class DataCleaner(object):
    """docstring for ClassName"""

    def __init__(self, city_name):
        self.city_name = city_name

    def clean_listing(self):
