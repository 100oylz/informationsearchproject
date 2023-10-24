from svd.FunkSVD import FunkSVD
from dataset import dataset
import pickle

data = dataset()
print(data.rawmatrixshape)
print(data.itemnum)
svd = FunkSVD(data.rawmatrixshape, 64, lr_all=1e-3, lambda_all=5e-4)
ratings = svd.fit(data, num_epochs=200, journal_path='journal/FunkSVD_model1.txt')
print(ratings.train_loss_list)

with open("funksvd.pkl", 'wb') as f:
    pickle.dump(ratings, f)
