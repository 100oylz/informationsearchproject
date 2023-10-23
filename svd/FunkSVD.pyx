import numpy as np
cimport numpy as np
from cython cimport list

class FunkSVD:

    # Add more cdef declarations for other variables

    def __init__(self, shape: tuple[int, int], hidden_classes: int = 20, init_mean=0, init_std=1,
                 lr_userbias=None, lr_itembias=None, lr_P=None, lr_Q=None, lambda_userbias=None, lambda_itembias=None,
                 lambda_P=None, lambda_Q=None, lambda_all=2e-2, lr_all=2e-2, verbose=False):
        assert len(shape) == 2, "Shape Dimension Must be 2!"

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
        self.MeanBias = 0.0
        self.train_loss_list=list()
        

    def fit(self, trainset, num_epochs=100):
        self.trainset = trainset
        self.sgd(trainset, num_epochs)
        return self

    def sgd(self, trainset, num_epochs):
        cdef np.ndarray[np.double_t,ndim=2] P,Q
        cdef np.ndarray[np.double_t,ndim=1] UserBias,ItemBias

        UserBias = np.zeros((self.shape[0]), dtype=np.double)
        ItemBias = np.zeros((self.shape[1]), dtype=np.double)
        P = np.random.normal(self.init_mean, self.init_std, (self.shape[0], self.hidden_classes))
        Q = np.random.normal(self.init_mean, self.init_std, (self.hidden_classes, self.shape[1]))

        for epoch in range(1, num_epochs + 1):
            total_err = 0
            for iteration, (u, i, r) in enumerate(trainset.all_ratings(), 1):
                dot = np.dot(P[u, :], Q[:, i])
                err = r - (ItemBias[i] + UserBias[u] + dot + self.MeanBias)

                # 每隔1000个迭代周期打印一次err
                if iteration % 1000 == 0:
                    print("Iteration:", iteration, "Error:", np.abs(err))
                UserBias[u] += self.lr_userbias * (err - self.lambda_userbias * UserBias[u])
                ItemBias[i] += self.lr_itembias * (err - self.lambda_itembias * ItemBias[i])

                for f in range(self.hidden_classes):
                    p_uf = P[u, f]
                    q_fi = Q[f, i]
                    P[u, f] += self.lr_P * (err * q_fi - self.lambda_P * p_uf)
                    Q[f, i] += self.lr_Q * (err * p_uf - self.lambda_Q * q_fi)
                total_err += np.sqrt(err ** 2)
            epoch_err = float(total_err) / trainset.itemnum
            print(f'Epoch {epoch}/{num_epochs}: loss->{epoch_err}')
            self.P=P
            self.Q=Q
            self.UserBias=UserBias
            self.ItemBias=ItemBias
            self.train_loss_list.append(epoch_err)
            

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
