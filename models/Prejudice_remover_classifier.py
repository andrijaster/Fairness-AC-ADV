from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover
import numpy as np

class Fair_PR():

    def __init__(self, sensitive, class_attr, eta):
        self.model = PrejudiceRemover(sensitive_attr = sensitive, class_attr=class_attr, eta=eta)
        
    def fit(self, data, labels, prot):
        ds = BinaryLabelDataset(df = data, label_names = labels, 
                             protected_attribute_names= prot)
        self.model.fit(ds)

    def predict_proba(self, data_test):
        y = self.model.predict(data_test)
        return y.scores