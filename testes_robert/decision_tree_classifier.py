from node import DTNode
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random


class DecisionTreeClassifier:
    def __init__(self, max_depth:int=None, min_samples_split:int=None, criterium:str='entropy') -> None:
        self.root:DTNode = None
        self.max_depth:int = max_depth 
        self.min_samples_split:int = min_samples_split 
        self.criterium:int = criterium


    def build_tree(self, dataset:DataFrame, curr_depth:int=0) -> DTNode:
        '''Construct the Decision Tree from the root node''' 
        X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples = X.shape[0]
        
        if num_samples<self.min_samples_split or curr_depth==self.max_depth or self.is_pure(y):
            return DTNode(leaf_value=self.calculate_leaf_value(y))
        
        best_split = self.get_best_split(dataset) 
        if best_split["info_gain"]==0: return DTNode(leaf_value=self.calculate_leaf_value(y)) # Ou é uma folha ou o split dividiu 50/50. Se o melhor split dividiu 50/50, é pq todos os splits possíveis são ou folhas ou dividem 50/50. Ambos os casos CREIO EU, NA MINHA CABEÇA, melhor retornar uma folha. No caso do 50/50, retorna um valor aleatorio.

        children = []
        for child in best_split["dataset_children"]:
            children.append(self.build_tree(child), curr_depth+1)
        return DTNode(best_split["feature_index"], best_split["feature_name"], best_split["threshold"], children, best_split["info_gain"])

    

    def get_best_split(self, dataset:DataFrame) -> dict:
        '''Get the best split for a node'''
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in dataset.shape[1]:
            feature_name = dataset.columns[feature_index]       
            feature_values = dataset.iloc[:, feature_index]    
            y = dataset.iloc[:, -1]
            if self.get_attr_type(feature_index) == 'binary': 
                split = self.binary_split(dataset, feature_name, feature_values, y)
            else: 
                split = self.multiclass_split(dataset, feature_name, feature_values, y)
            if split is None: continue
         
           
            curr_info_gain = self.information_gain(y, split[1])  # split: threshold + divisões de dataset geradas no split
            if curr_info_gain>max_info_gain:
                best_split["feature_index"] = feature_index
                best_split["feature_name"] = feature_name
                best_split["threshold"] = split[0]
                best_split["dataset_children"] = split[1]
                best_split["info_gain"] = curr_info_gain
                max_info_gain = curr_info_gain    
        return best_split

    def binary_split(self, dataset, feature_name, feature_values):
        best_division = None
        max_info_gain = -float("inf")
        possible_thresholds = pd.unique(feature_values)
        for threshold in possible_thresholds:
            y = dataset.iloc[:, -1]
            children = self.split(dataset, feature_name, threshold) 
            curr_info_gain = self.information_gain(y, children)
            if curr_info_gain>max_info_gain:
                best_division = children
                max_info_gain = curr_info_gain
        return max_info_gain, best_division
    
    def multiclass_split(self, dataset, feature_name, feature_values, y):
        children = []
        for value in feature_values:
            df_children = dataset.query(f"{feature_name} == {value}")
            children.append(df_children)
            curr_info_gain = self.information_gain(y, df_children)
        return None, children
    

    def is_pure(self, target_column:Series) -> bool:
        target_column = set(target_column)
        return len(target_column) == 1


    def info_gain_binary(self, y_parent:Series, y_left:Series, y_right:Series) -> float:
        '''Get the information gain for a node'''
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        if self.criterium=="gini":
            gain = self.gini_index(y_parent) - (weight_left*self.gini_index(y_left) + weight_right*self.gini_index(y_right))
        else:
            gain = self.entropy(y_parent) - (weight_left*self.entropy(y_left) + weight_right*self.entropy(y_right))
        return gain
    
    def info_gain_multiclass(self, y_parent:Series, children, y_parent_index) -> float:
        '''Get the information gain for a node'''
        entropies = []
        for child in children:
            y_child = child.iloc[:,y_parent_index]
            child_weight = len(child) / len(y_parent)
            if self.criterium=="gini":
                intern_entropy = child_weight*self.gini_index(y_child)
            else:
                intern_entropy = child_weight*self.entropy(y_child)
            entropies.append(intern_entropy)
        if self.criterium=="gini":
            return  self.gini_index(y_parent) - sum(entropies)
        return self.entropy(y_parent) - sum(entropies)
    

    def entropy(self, y:Series) -> float:
        '''Get the entropy value for a node'''
        target_column = pd.unique(y)
        entropy = 0
        for attribute in target_column:
            p_cls = len(y[y == attribute]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    

    def gini_index(self, y:Series) -> float:
        '''Get the gini value for a node'''
        class_labels = pd.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        

    # def calculate_leaf_value(self, y:Series) -> any:
    #     '''Get the majority of results in a leaf node'''
    #     list_y = list(y)
    #     return max(set(list_y), key=list_y.count)
    
    def calculate_leaf_value(self, y: Series) -> any:
        '''Get the majority of results in a leaf node'''
        list_y = list(y)
        max_count = max(list_y.count(item) for item in set(list_y)) # Máximo contador da lista
        most_common_values = [item for item in set(list_y) if list_y.count(item) == max_count] # Cria lista com valores com contador máximo
        return random.choice(most_common_values)
    

    def fit(self, X:DataFrame, y:Series) -> None:
        '''Fit the tree with a csv for trainning'''
        dataset = pd.concat((X, y), axis=1)
        self.root = self.build_tree(dataset)
    

    def get_attr_type(value):
        return 'multiclass'
    
    # def predict(self, X_test:DataFrame) -> list[any]:
    #     '''Predict results based on the trained tree and a new dataframe test'''
    #     samples, _ = X_test.shape
    #     predictions = []
    #     for row_index in range(samples):
    #         csv_row = X_test.iloc[row_index]
    #         predictions.append(self.make_prediction(csv_row, self.root, X_test))
    #     return predictions
    

    # def make_prediction(self, csv_row: tuple, node:DTNode, X_test:DataFrame) -> any:
    #     '''Predict prediction for each row in dataframe'''
    #     if node.leaf_value is not None: return node.leaf_value
    #     feature_val = csv_row.iloc[node.feature_index]
    #     if self.get_attr_type(feature_val) == 'multiclass':
    #         for value in X_test.iloc[:,node.feature_index]:
    #             return self.make_prediction(csv_row, node.left_node, X_test)
    #     else:
    #         if feature_val==PRIMEIRO_VALOR
    #             return self.make_prediction(csv_row, node.left_node, X_test)
    #         return self.make_prediction(csv_row, node.right_node, X_test)
    

    