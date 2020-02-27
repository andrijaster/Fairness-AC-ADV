from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.preprocessing import Reweighing

class Fair_rew_RF():
    
    def __init__(self, un_gr, pr_gr, n_est = 100, min_sam_leaf = 25):
        self.model_reweight = Reweighing(un_gr,pr_gr)
        self.model = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_sam_leaf)
        
    def fit(self, data, labels, prot):
        ds = BinaryLabelDataset(df = data, label_names = labels, 
                             protected_attribute_names= prot)
        x = self.model_reweight.fit_transform(ds)
        self.model.fit(x.features, x.labels.ravel())

    def predict_proba(self, data_test):
        x = self.model_reweight.transform(data_test)
        y = self.model.predict_proba(x.features)[:,1]
        return y
        
