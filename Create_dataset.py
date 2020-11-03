import pandas as pd
import numpy as np
import os
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

from aif360.datasets import AdultDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import GermanDataset



def Adult_dataset(name_prot = 'sex'):
    dataset_orig = AdultDataset(protected_attribute_names=['sex'],
            privileged_classes= [['Male']],
            features_to_keep=['age', 'education-num'])
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'income-per-year':'labels'}, inplace = True)
    data.reset_index(inplace = True, drop = True)
    data.to_csv("dataset/Adult.csv")


def Readmission():
    folder_name = os.path.join('datasets_raw','readmission.csv')
    data = pd.read_csv(folder_name)
    data.drop(['ID','readmitDAYS'], axis = 1, inplace = True)
    data.rename(columns={'readmitBIN':'labels'}, inplace = True)
    data.to_csv("dataset/Readmission.csv")

def Medical_EXPEN_dataset(name_prot = 'RACE'):
    dataset_orig = MEPSDataset19()
    data, _ = dataset_orig.convert_to_dataframe()
    data.reset_index(inplace = True, drop = True)
    output = dataset_orig.labels
    out = pd.DataFrame(output, columns = ["labels"])
    data = pd.concat([data,out],axis = 1, join = 'inner')
    data.to_csv("dataset/Medical_expendetures.csv")

def german_dataset_age(name_prot=['age']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,
        privileged_classes=[lambda x: x >= 25],      
        features_to_drop=['personal_status', 'sex'])
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'credit':'labels'}, inplace = True)
    data.to_csv("dataset/German_age.csv")



def german_dataset_sex(name_prot=['sex']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,                                                               
        features_to_drop=['personal_status', 'age'])
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'credit':'labels'}, inplace = True)
    data.to_csv("dataset/German_sex.csv")


if __name__ == "__main__":
    Adult_dataset()
    Readmission()
    Medical_EXPEN_dataset()
    german_dataset_age()
    german_dataset_sex()

    df1 = pd.read_csv("dataset/Adult.csv")
    df2 = pd.read_csv("dataset/Readmission.csv")
    df3 = pd.read_csv("dataset/Medical_expendetures.csv")
    df4 = pd.read_csv("dataset/German_age.csv")
    df5 = pd.read_csv("dataset/German_sex.csv")