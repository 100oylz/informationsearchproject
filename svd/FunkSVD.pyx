import numpy as np
cimport numpy as np
from cython cimport list
import tqdm

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


    def fit(self, trainset, num_epochs=100, preload=False, journal_path=None):
        self.trainset = trainset
        self.sgd(trainset, num_epochs,preload,journal_path)
        return self

    def sgd(self, trainset, num_epochs, preload=False, journal_path=None):
        cdef np.ndarray[np.double_t,ndim=2] P,Q
        cdef np.ndarray[np.double_t,ndim=1] UserBias,ItemBias

        UserBias = np.zeros((self.shape[0]), dtype=np.double)
        ItemBias = np.zeros((self.shape[1]), dtype=np.double)
        P = np.random.normal(self.init_mean, self.init_std, (self.shape[0], self.hidden_classes))
        Q = np.random.normal(self.init_mean, self.init_std, (self.hidden_classes, self.shape[1]))

        # 打开日志文件
        if journal_path:
            if preload:
                mode = "a"  # 追加模式
            else:
                mode = "w"  # 覆写模式
            journal_file = open(journal_path, mode)
        else:
            journal_file = None

        for epoch in range(1, num_epochs + 1):
            total_err = 0
            total_valid_err=0
            epoch_bar = tqdm.tqdm(trainset.all_train_set(), total=trainset.train_set_item)
            epoch_bar.set_description(f"Epoch {epoch}/{num_epochs}")
            for (u, i, r) in epoch_bar:
                dot = np.dot(P[u, :], Q[:, i])
                err = r - (ItemBias[i] + UserBias[u] + dot + self.MeanBias)

                UserBias[u] += self.lr_userbias * (err - self.lambda_userbias * UserBias[u])
                ItemBias[i] += self.lr_itembias * (err - self.lambda_itembias * ItemBias[i])

                for f in range(self.hidden_classes):
                    p_uf = P[u, f]
                    q_fi = Q[f, i]
                    P[u, f] += self.lr_P * (err * q_fi - self.lambda_P * p_uf)
                    Q[f, i] += self.lr_Q * (err * p_uf - self.lambda_Q * q_fi)
                total_err += err ** 2
            epoch_err = float(total_err) / trainset.itemnum
            print(f'Epoch {epoch}/{num_epochs}: loss->{epoch_err}')

            # 记录打印内容到日志文件
            if journal_file:
                journal_file.write(f'Epoch {epoch}/{num_epochs}: loss->{epoch_err}\n')
            epoch_bar=tqdm.tqdm(trainset.all_valid_set(),total=trainset.validation_set_item)
            epoch_bar.set_description(f"Epoch {epoch}/{num_epochs}")
            for u,i,r in epoch_bar:
                dot = np.dot(P[u, :], Q[:, i])
                err = r - (ItemBias[i] + UserBias[u] + dot + self.MeanBias)
                total_valid_err+=err**2
            valid_epoch_err=float(total_valid_err)/trainset.validation_set_item
            print(f'Epoch {epoch}/{num_epochs}: valid_loss->{valid_epoch_err}')

            # 记录打印内容到日志文件
            if journal_file:
                journal_file.write(f'Epoch {epoch}/{num_epochs}: valid_loss->{valid_epoch_err}\n')
            self.P=P
            self.Q=Q
            self.UserBias=UserBias
            self.ItemBias=ItemBias
            self.train_loss_list.append(epoch_err)

        # 关闭日志文件
        if journal_file:
            journal_file.close()
            

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

    def estimatetestset(self):
        if(self.trainset):
            total_err=0.0
            for u,i,r in self.trainset.all_test_set():
                dot = np.dot(self.P[u, :], self.Q[:, i])
                err = r - (self.ItemBias[i] + self.UserBias[u] + dot + self.MeanBias)
                total_err+=err**2
            test_err=float(total_err)/self.trainset.test_set_item
            print(f'Test Set Loss->{test_err}')
            

    def predict(self, userid, itemid):
        est = self.estimate(userid, itemid)
        est -= self.trainset.offset
        return est
