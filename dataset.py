import numpy as np
import utils
from constant import *
from sklearn.model_selection import train_test_split


class dataset():
    def __init__(self, offset=0.0):
        self.offset = offset
        ratings = utils.loadcsv(SMALLRATINGSPATH)
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

    def all_valid_set(self):
        for u, i, r in self.validation_set:
            yield u, i, r

    def knows_user(self, id):
        return id in self.user_set

    def knows_item(self):
        return id in self.movie_set

    def global_mean(self):
        if (self._global_mean == None):
            self._global_mean = np.mean(self.grades)
        return self._global_mean

    def split_train_test(self, randomstate=1):

        # 使用train_test_split将数据集划分为训练集（80%）和临时集（20%）
        train_users, temp_users, train_movies, temp_movies, train_grades, temp_grades = train_test_split(self.users,
                                                                                                         self.movies,
                                                                                                         self.grades,
                                                                                                         test_size=0.2,
                                                                                                         shuffle=True,
                                                                                                         random_state=randomstate)

        # 将临时集按照1:1的比例划分为验证集和测试集
        test_users, val_users, test_movies, val_movies, test_grades, val_grades = train_test_split(temp_users,
                                                                                                   temp_movies,
                                                                                                   temp_grades,
                                                                                                   test_size=0.5,
                                                                                                   shuffle=True,
                                                                                                   random_state=randomstate)

        # 创建训练集、验证集和测试集
        self.train_set = list(zip(train_users, train_movies, train_grades))
        self.validation_set = list(zip(val_users, val_movies, val_grades))
        self.test_set = list(zip(test_users, test_movies, test_grades))

        # 计算数据集大小
        self.train_set_item = len(self.train_set)
        self.validation_set_item = len(self.validation_set)
        self.test_set_item = len(self.test_set)

        print("训练集大小：", self.train_set_item)
        print("验证集大小：", self.validation_set_item)
        print("测试集大小：", self.test_set_item)


if __name__ == '__main__':
    data = dataset()
