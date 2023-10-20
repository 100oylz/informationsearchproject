import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from constant import MOVIESDPATH, RATINGSPATH


def loadcsv(filename):
    print(f'load {filename}')
    data = pd.read_csv(filename)
    return data


def preprocessingData(movies: pd.DataFrame, ratings: pd.DataFrame):
    row_indices = ratings['userId'].values
    col_indices = ratings['movieId'].values
    data = ratings['rating'].values
    sparse_data = sp.csr_matrix((data, (row_indices, col_indices)))
    # print(sparse_data)
    return sparse_data


if __name__ == '__main__':
    # movies:movieId,title,genres
    # ratings:userId,movieId,rating,timestamp
    movies, ratings = loadcsv(MOVIESDPATH), loadcsv(RATINGSPATH)

    preprocessingData(movies, ratings)
