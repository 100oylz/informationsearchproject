import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class FunkSVD(nn.Module):
    def __init__(self, shape, hidden_classes=20, init_mean=0, init_std=1,
                 lr_userbias=None, lr_itembias=None, lr_P=None, lr_Q=None, lambda_userbias=None, lambda_itembias=None,
                 lambda_P=None, lambda_Q=None, lambda_all=2e-2, lr_all=2e-2, verbose=False):
        super(FunkSVD, self).__init__()
        print("Pytorch")
        assert len(shape) == 2, "Shape Dimension Must be 2!"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Use Cuda")
        self.shape = shape
        self.hidden_classes = hidden_classes
        self.verbose = verbose
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_userbias = lr_userbias if lr_userbias is not None else lr_all
        self.lr_itembias = lr_itembias if lr_itembias is not None else lr_all
        self.lr_P = lr_P if lr_P is not None else lr_all
        self.lr_Q = lr_Q if lr_Q is not None else lr_all
        self.lambda_userbias = lambda_userbias if lambda_userbias is not None else lambda_all
        self.lambda_itembias = lambda_itembias if lambda_itembias is not None else lambda_all
        self.lambda_P = lambda_P if lambda_P is not None else lambda_all
        self.lambda_Q = lambda_Q if lambda_Q is not None else lambda_all
        self.MeanBias = 0.0
        self.train_loss_list = []

        self.UserBias = nn.Parameter(torch.zeros(shape[0]))
        self.ItemBias = nn.Parameter(torch.zeros(shape[1]))
        self.P = nn.Parameter(torch.empty(shape[0], hidden_classes).normal_(init_mean, init_std))
        self.Q = nn.Parameter(torch.empty(hidden_classes, shape[1]).normal_(init_mean, init_std))

    def forward(self, user_indices, item_indices):
        dot = torch.matmul(self.P[user_indices], self.Q[:, item_indices])
        return self.ItemBias[item_indices] + self.UserBias[user_indices] + dot + self.MeanBias

    def fit(self, trainset, num_epochs=100):
        self.trainset = trainset
        self.sgd(trainset, num_epochs)
        return self

    def sgd(self, trainset, num_epochs):
        param_groups = [
            {'params': self.P, 'lr': self.lr_P},
            {'params': self.Q, 'lr': self.lr_Q},
            {'params': self.UserBias, 'lr': self.lr_userbias},
            {'params': self.ItemBias, 'lr': self.lr_itembias},
        ]
        optimizer = optim.SGD(param_groups, lr=1e-3, weight_decay=5e-4)
        criterion = nn.MSELoss()
        self.to(self.device)

        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            epoch_bar = tqdm.tqdm(trainset.all_train_set(), total=trainset.itemnum)
            for (u, i, r) in epoch_bar:
                r = torch.tensor(r)
                r = r.float()
                r = r.to(self.device)
                optimizer.zero_grad()
                est = self.forward(u, i)
                # print(est, r)
                assert r.shape == est.shape, f"Shpae Different,{est.shape}!={r.shape}"
                loss = criterion(est, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / trainset.itemnum
            print(f'Epoch {epoch}/{num_epochs}: loss->{epoch_loss}')
            self.train_loss_list.append(epoch_loss)

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
            est += torch.dot(self.P[u, :], self.Q[:, i])
        return est.item()

    def predict(self, userid, itemid):
        est = self.estimate(userid, itemid)
        est -= self.trainset.offset
        return est
