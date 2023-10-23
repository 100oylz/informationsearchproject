import numpy as np
import utils
from constant import *


class dataset():
    def __init__(self, offset=0.0):
        self.offset = offset
        ratings = utils.loadcsv(RATINGSPATH)
        self.users = ratings['userId'].values
        self.movies = ratings['movieId'].values
        self.grades = ratings['rating'].values
        self._global_mean = None
        self.user_set = set(self.users)
        self.movie_set = set(self.movies)
        self.itemnum = len(self.grades)
        self.rawmatrixshape = (max(self.user_set), max(self.movie_set))

    def all_ratings(self):
        for u, i, r in zip(self.users, self.movies, self.grades):
            yield u, i, r

    def knows_user(self, id):
        return id in self.user_set

    def knows_item(self):
        return id in self.movie_set

    def global_mean(self):
        if (self._global_mean == None):
            self._global_mean = np.mean(self.grades)
        return self._global_mean
