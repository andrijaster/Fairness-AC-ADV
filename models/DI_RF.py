from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.preprocessing import DisparateImpactRemover
import numpy as np

class Fair_DI_RF():
    
    def __init__(self, sensitive, repair_level, n_est = 100, min_sam_leaf = 25):
        if repair_level<0:
            repair_level = 0
        elif repair_level>1:
            repair_level = 1
        self.model_reweight = DisparateImpactRemover(sensitive_attribute = sensitive, repair_level=repair_level)
        self.model = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_sam_leaf)
        
    def fit(self, data, labels, prot):
        ds = BinaryLabelDataset(df = data, label_names = labels, 
                             protected_attribute_names= prot)
        self.prot = prot
        x = self.model_reweight.fit_transform(ds)
        index = x.feature_names.index(prot[0])
        x_train = np.delete(x.features,index,1)
        y_train = x.labels.ravel()
        self.model.fit(x_train, y_train)

    def predict_proba(self, data_test):
        x = self.model_reweight.fit_transform(data_test)
        index = x.feature_names.index(self.prot[0])
        x_test = np.delete(x.features,index,1)
        y = self.model.predict_proba(x_test)[:,1]
        return y