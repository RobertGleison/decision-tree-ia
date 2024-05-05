from typing import Union
import numpy as np
import pandas as pd
import random
from decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from pandas import DataFrame, Series

class DecisionTree:

    def __init__(self, df: DataFrame, min_samples_split=5, max_depth=10) -> None:
        self.decisiontree = DecisionTreeModel(max_depth, min_samples_split, criterium='entropy')
        self.dataframe = df
        self.target_data = df.iloc[:,-1]
        self.features_data = df.iloc[:,:-1]
        self.features_names = self.features_data.columns

    def fit(self):
        self.decisiontree.fit(self.features_data, self.target_data)

    def predict(self):
        print("\n\nPREDICTION ---------")
        X_test = []
        for feature in self.features_names:
            feature_value = input(feature + "? ")
            if feature_value.isdigit():
                X_test.append(float(feature_value))
                continue
            if feature_value.upper() == 'FALSE': 
                X_test.append("False")
                continue
            if feature_value.upper() == 'TRUE': 
                X_test.append("True")
                continue
            X_test.append(feature_value)
        
        test = pd.DataFrame([X_test], columns=self.features_names)
        result = self.decisiontree.predict(test)[0]
        print("\nPREDICTION: ", result)

        
