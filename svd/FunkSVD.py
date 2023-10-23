import numpy as np


class FunkSVD():
    def __init__(self, shape: tuple[int, int], hidden_classes: int = 20, init_mean=0, init_std=1,
                 lr_userbias=None, lr_itembias=None, lr_P=None, lr_Q=None, lambda_userbias=None, lambda_itembias=None,
                 lambda_P=None, lambda_Q=None, lambda_all=2e-2, lr_all=2e-2, verbose=False):
        assert len(shape) == 2, "Shpae Dimension Must be 2!"
        self.shape = shape
        self.hidden_classes = hidden_classes
        self.verbose = verbose
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_userbias = lr_userbias if lr_userbias != None else lr_all
        self.lr_itembias = lr_itembias if lr_itembias != None else lr_all
        self.lr_P = lr_P if lr_P != None else lr_all
        self.lr_Q = lr_Q if lr_Q != None else lr_all
        self.lambda_userbias = lambda_userbias if lambda_userbias != None else lambda_all
        self.lambda_itembias = lambda_itembias if lambda_itembias != None else lambda_all
        self.lambda_P = lambda_P if lambda_P != None else lambda_all
        self.lambda_Q = lambda_Q if lambda_Q != None else lambda_all

    def fit(self, trainset):
        self.trainset = trainset
        self.sgd(trainset)
        return self

    def sgd(self, trainset, num_epochs):
        self.UserBias = np.zeros((self.shape[0], 1))
        self.ItemBias = np.zeros((1, self.shape[1]))
        self.MeanBias = 0
        self.P = np.random.normal(self.init_mean, self.init_std, (self.shape[0], self.hidden_classes))
        self.Q = np.random.normal(self.init_mean, self.init_std, (self.hidden_classes, self.shape[1]))
        self.train_loss_list = []
        for epoch in range(1, num_epochs + 1):
            total_err = 0
            for u, i, r in trainset.all_ratings():
                dot = np.dot(self.Q[i, :], self.P[:, u])
                err = r - (self.ItemBias[i] + self.UserBias[u] + dot + self.MeanBias)

                self.UserBias[u] += self.lr_userbias * (err - self.lambda_userbias * self.UserBias[u])
                self.ItemBias[i] += self.lr_itembias * (err - self.lambda_itembias * self.ItemBias[i])

                for f in range(self.hidden_classes):
                    p_uf = self.P[u, f]
                    q_fi = self.Q[f, i]
                    self.P[u, f] += self.lr_P * (err * q_fi - self.lambda_P * p_uf)
                    self.Q[f, i] += self.lr_Q * (err * p_uf - self.lambda_Q * q_fi)
                total_err += err
            self.train_loss_list.append(float(total_err) / trainset.itemnum)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        est = self.trainset.global_mean()
        if not (known_user and known_item):
            raise ValueError("User and Item are unknown!")
        if known_user:
            est += self.UserBias[u]
        if known_item:
            est += self.ItemBias[i]
        if known_item and known_user:
            est += np.dot(self.P[u, :], self.Q[:, i])
        return est

    def predict(self, userid, itemid):
        est = self.estimate(userid, itemid)
        est -= self.trainset.offset
        return est
