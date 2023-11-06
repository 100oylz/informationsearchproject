from svd.FunkSVD import FunkSVD
import pickle
with open('funksvd_1.pkl','rb') as f:
    model:FunkSVD=pickle.load(f)
print(model.estimatetestset())