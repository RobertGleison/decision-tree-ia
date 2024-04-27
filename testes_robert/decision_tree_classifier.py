from typing import Union
from node import DTNode
import numpy as np
import pandas as pd
import random
from pandas import DataFrame, Series


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = None, min_samples_split: int = None, criterium: str = 'entropy') -> None:
        self.root: DTNode = None
        self.max_depth: int = max_depth 
        self.min_samples_split: int = min_samples_split 
        self.criterium: str = criterium

    

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        '''Fit the tree with a DataFrame for trainning'''
        dataset = pd.concat((X_train, y_train), axis=1)
        self.root = self.build_tree(dataset)



    def build_tree(self, dataset: DataFrame, curr_depth: int = 0) -> DTNode:
        '''Construct the Decision Tree from the root node''' 
        X_train, y_train = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples = X_train.shape[0]
        
        if num_samples<self.min_samples_split or curr_depth==self.max_depth or self.is_pure(y_train): # Usei um early return aq
            return DTNode(leaf_value=self.calculate_leaf_value(y_train))
        
        best_split = self.get_best_split(dataset) 
        if best_split["info_gain"]==0: return DTNode(leaf_value=self.calculate_leaf_value(y_train)) # Ou é uma folha ou o split dividiu 50/50. Se o melhor split dividiu 50/50, é pq todos os splits possíveis são ou folhas ou dividem 50/50. Ambos os casos CREIO EU, NA MINHA CABEÇA, melhor retornar uma folha. No caso do 50/50, retorna um valor aleatorio.

        children = []
        for child in best_split["children"]:
            children.append(self.build_tree(child, curr_depth+1))
        return DTNode(feature_index=["feature_index"], feature_name=best_split["feature_name"], children=best_split["children"], info_gain=best_split["info_gain"], split_values=best_split["split_values"])


    
    def calculate_leaf_value(self, y_train: Series) -> any: # Antes só funcionava pra folhas puras, agr funciona pra folhas 50/50 caso o info gain seja 0
        '''Get value of the majority of results in a leaf node'''
        list_y = list(y_train)
        max_count = max(list_y.count(item) for item in set(list_y)) # Máximo contador da lista
        most_common_values = [item for item in set(list_y) if list_y.count(item) == max_count] # Cria lista com valores com contador máximo
        return random.choice(most_common_values)
    


    def is_pure(self, target_column: Series) -> bool:
        '''Check if a node have only one type of target value'''
        return len(set(target_column)) == 1
    


    def get_best_split(self, dataset: DataFrame) -> dict:
        '''Get the best split for a node'''
        best_split = {}
        max_info_gain = 0
        y_train = dataset.iloc[:, -1]
        train_columns = dataset.shape[1] - 1 

        for feature_index in range(train_columns):
            feature_name = dataset.columns[feature_index]       
            values = dataset.iloc[:, feature_index]    
            attr_type = self.get_attr_type(dataset, feature_name)

            if attr_type == 'multiclass_discrete': 
                children, info_gain = self.multiclass_discrete_split(dataset, feature_index, pd.unique(values), y_train)
                max_info_gain = self.update_best_split(best_split, info_gain, max_info_gain, children, values, feature_name, feature_index)
                continue

            for value in pd.unique(values):
                if attr_type == 'binary_discrete': children, info_gain = self.binary_discrete_split(dataset, feature_index, value, y_train)
                if attr_type == 'continuous': children, info_gain = self.multiclass_discrete_split(dataset, feature_index, value, y_train)

                if children is None: continue
                max_info_gain = self.update_best_split(best_split, info_gain, max_info_gain, children, values, feature_name, feature_index)
        return best_split
    


    def update_best_split(self, best_split: dict, info_gain: float, max_info_gain: float, children: list, value: any, feature_name: str, feature_index: int) -> float:
        if info_gain > max_info_gain:
            best_split["feature_index"] = feature_index
            best_split["feature_name"] = feature_name
            best_split["split_values"] = value
            best_split["children"] = children
            best_split["info_gain"] = info_gain
            return info_gain
        return max_info_gain



    def get_attr_type(self, dataset: DataFrame, feature_name: str) -> str: 
        if type(dataset.iloc[0][feature_name]) == int or type(dataset.iloc[0][feature_name]) == float: return 'continuous'
        unique_values = pd.unique(dataset[feature_name])
        if len(unique_values) > 2: return 'multiclass_discrete'
        return 'binary_discrete'


    def continuous_split(self, dataset: DataFrame, feature_index: int, threshold: any, y_parent: Series) -> tuple[list, float]:
        '''Split the DataFrame with a continuous value'''
        left = dataset[dataset.iloc[:, feature_index] <= threshold]
        right = dataset[dataset.iloc[:, feature_index] > threshold]
        children = [left, right]
        info_gain = self.binary_info_gain(y_parent, left.iloc[:,-1], right.iloc[:,-1])
        return children, info_gain
    


    def binary_discrete_split(self, dataset: DataFrame, feature_index: int, value: any, y_parent: Series) -> tuple[list, float]:
        '''Split the DataFrame with a discrete and binary value'''
        left = dataset[dataset.iloc[:, feature_index] == value]
        right = dataset[dataset.iloc[:, feature_index] != value]
        children = [left, right]
        info_gain = self.binary_info_gain(y_parent, left.iloc[:,-1], right.iloc[:,-1])
        return children, info_gain



    def multiclass_discrete_split(self, dataset: DataFrame, feature_index: int, values: Series, y_parent: Series) -> tuple[list, float]:
        '''Split the DataFrame with a discrete and multiclass value'''
        labels = pd.unique(values)
        children = []
        for label in labels:
            child_dataset = dataset[dataset.iloc[:, feature_index] == label]
            children.append(child_dataset)
        info_gain = self.info_gain_multiclass(y_parent, children)
        return children, info_gain



    def info_gain_multiclass(self, y_parent: Series, children: list) -> float:
        '''Get the information gain for a node splitted by a multiclass discrete value'''
        children_impurity_sum = 0
        for child in children:
            children_impurity_sum += len(child) / len(y_parent) * self.get_impurity(child.iloc[:, -1])
        return self.get_impurity(y_parent) - children_impurity_sum



    def get_impurity(self, y_train: Series) -> float:
        '''Get the impority of the node'''
        return self.gini_index(y_train) if self.criterium=='gini' else self.entropy(y_train)



    def entropy(self, y_train: Series) -> float:
        '''Get the entropy value for a node'''
        class_labels = pd.unique(y_train)
        entropy = 0
        for label in class_labels:
            p_cls = len(y_train[y_train == label]) / len(y_train)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    


    def gini_index(self, y_train: Series) -> float:
        '''Get the gini value for a node'''
        class_labels = pd.unique(y_train)
        gini = 0
        for label in class_labels:
            p_cls = len(y_train[y_train == label]) / len(y_train)
            gini += p_cls**2
        return 1 - gini



    def binary_info_gain(self, y_parent: Series, y_left: Series, y_right: Series) -> float:
        '''Get the information gain for a node splitted by a binary or threshold value'''
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        return self.get_impurity(y_parent) - weight_left*self.get_impurity(y_left) + weight_right*self.get_impurity(y_right)
    


    def predict(self, X_test: DataFrame) -> list:
        '''Predict target column for a dataframe'''
        return [self.make_prediction(row, self.root, X_test) for _, row in X_test.iterrows()]



    def make_prediction(self, row: tuple, node: DTNode, X_test: DataFrame) -> Union[any, None]:
        '''Predict target for each row in dataframe'''
        if node.leaf_value is not None: 
            return node.leaf_value
        
        attr_type = self.get_attr_type(X_test, node.feature_name)
        value = row[node.feature_index]  # Accessing the feature value from the row directly

        if attr_type == 'multiclass_discrete' or attr_type == 'binary_discrete': 
            for i, node_value in enumerate(node.split_values):
                if value == node_value:
                    return self.make_prediction(row, node.children[i], X_test)  

        elif attr_type == 'continuous':
            if value <= node.split_values[0]:
                return self.make_prediction(row, node.children[0], X_test)  
            return self.make_prediction(row, node.children[1], X_test)

        return None

    

    