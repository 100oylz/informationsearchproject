import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from constant import *


def loadcsv(filename):
    print(f'load {filename}')
    data = pd.read_csv(filename)
    return data


if __name__ == '__main__':
    # movies:movieId,title,genres
    # ratings:userId,movieId,rating,timestamp
    movies, ratings = loadcsv(MOVIESPATH), loadcsv(RATINGSPATH)

