import numpy as np
import utils
from constant import *
from sklearn.model_selection import train_test_split


class dataset():
    def __init__(self, offset=0.0):
        self.offset = offset
        ratings = utils.loadcsv(RATINGSPATH)
        self.users = ratings['userId'].values - 1
        self.movies = ratings['movieId'].values - 1
        self.grades = ratings['rating'].values
        self._global_mean = None
        self.user_set = set(self.users)
        self.movie_set = set(self.movies)
        self.itemnum = len(self.grades)
        self.rawmatrixshape = (max(self.user_set) + 1, max(self.movie_set) + 1)
        self.split_train_test()

    def all_train_set(self):
        for u, i, r in self.train_set:
            yield u, i, r

    def all_test_set(self):
        for u, i, r in self.test_set:
            yield u, i, r

    def knows_user(self, id):
        return id in self.user_set

    def knows_item(self):
        return id in self.movie_set

    def global_mean(self):
        if (self._global_mean == None):
            self._global_mean = np.mean(self.grades)
        return self._global_mean

    def split_train_test(self):

        train_users, test_users, train_movies, test_movies, train_grades, test_grades = train_test_split(self.users,
                                                                                                         self.movies,
                                                                                                         self.grades,
                                                                                                         test_size=0.2,
                                                                                                         shuffle=True)
        self.train_set = list(zip(train_users, train_movies, train_grades))
        self.test_set = list(zip(test_users, test_movies, test_grades))
        self.train_set_item = len(self.train_set)
        self.test_set_item = len(self.test_set)
        print("训练集大小：", self.train_set_item)
        print("测试集大小：", self.test_set_item)


if __name__ == '__main__':
    data = dataset()
