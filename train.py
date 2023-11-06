from svd.FunkSVD import FunkSVD
from dataset import dataset
import pickle

data = dataset()
print(data.rawmatrixshape)
print(data.itemnum)

learning_rates = [1e-2]
lambda_values = [5e-2]
num_epoches=[50]

for i in range(len(learning_rates)):
    svd = FunkSVD(data.rawmatrixshape, 32, lr_all=learning_rates[i], lambda_all=lambda_values[i])
    ratings = svd.fit(data, num_epochs=num_epoches[i], journal_path=f'journal/FunkSVD_model{i+1}.txt')
    ratings.estimatetestset()
    # print(ratings.train_loss_list)

    with open(f"funksvd_{i+1}.pkl", 'wb') as f:
        pickle.dump(ratings, f)
