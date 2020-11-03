from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
import numpy as np

class Fair_meta():

    def __init__(self, sensitive, tau):
        self.model = MetaFairClassifier(sensitive_attr = sensitive, tau=tau)
        
    def fit(self, data, labels, prot):
        ds = BinaryLabelDataset(df = data, label_names = labels, 
                             protected_attribute_names= prot)
        self.model.fit(ds)

    def predict_proba(self, data_test):
        y = self.model.predict(data_test)
        return y.scores