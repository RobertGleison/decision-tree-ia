from node import DTNode
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random


class DecisionTreeClassifier:
    def __init__(self, max_depth:int = None, min_samples_split:int = None, criterium:str = 'entropy') -> None:
        self.root:DTNode = None
        self.max_depth:int = max_depth 
        self.min_samples_split:int = min_samples_split 
        self.criterium:int = criterium

    

    def fit(self, X:DataFrame, y:Series) -> None:
        '''Fit the tree with a csv for trainning'''
        dataset = pd.concat((X, y), axis=1)
        self.root = self.build_tree(dataset)



    def build_tree(self, dataset:DataFrame, curr_depth:int = 0) -> DTNode:
        '''Construct the Decision Tree from the root node''' 
        X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples = X.shape[0]
        
        if num_samples<self.min_samples_split or curr_depth==self.max_depth or self.is_pure(y): # Usei um early return aq
            return DTNode(leaf_value=self.calculate_leaf_value(y))
        
        best_split = self.get_best_split(dataset) 
        if best_split["info_gain"]==0: return DTNode(leaf_value=self.calculate_leaf_value(y)) # Ou é uma folha ou o split dividiu 50/50. Se o melhor split dividiu 50/50, é pq todos os splits possíveis são ou folhas ou dividem 50/50. Ambos os casos CREIO EU, NA MINHA CABEÇA, melhor retornar uma folha. No caso do 50/50, retorna um valor aleatorio.

        children = []
        for child in best_split["dataset_children"]:
            children.append(self.build_tree(child), curr_depth+1)
        return DTNode(best_split["feature_index"], best_split["feature_name"], best_split["threshold"], children, best_split["info_gain"])


    
    def calculate_leaf_value(self, y: Series) -> any: # Antes só funcionava pra folhas puras, agr funciona pra folhas 50/50 caso o info gain seja 0
        '''Get the majority of results in a leaf node'''
        list_y = list(y)
        max_count = max(list_y.count(item) for item in set(list_y)) # Máximo contador da lista
        most_common_values = [item for item in set(list_y) if list_y.count(item) == max_count] # Cria lista com valores com contador máximo
        return random.choice(most_common_values)
    


    def is_pure(self, target_column:Series) -> bool:
        return len(set(target_column)) == 1
    


    def get_best_split(self, dataset:DataFrame) -> dict:
        '''Get the best split for a node'''
        best_split = {}
        max_info_gain = 0
        y = dataset.iloc[:, -1]
        train_columns = dataset.shape[1] - 1 

        for feature_index in range(train_columns):
            feature_name = dataset.columns[feature_index]       
            feature_values = dataset.iloc[:, feature_index]    
            attr_type = self.get_attr_type(dataset, feature_name)

            if attr_type == 'multiclass_discrete': children, info_gain = self.multiclass_discrete_split(dataset, feature_index, feature_values, y)

            for value in pd.unique(feature_values):
                if attr_type == 'binary_discrete': children, info_gain = self.binary_discrete_split(dataset, feature_index, value, y)
                if attr_type == 'continuous': children, info_gain = self.multiclass_discrete_split(dataset, feature_index, value, y)

                if children is None: continue
                if info_gain>max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["feature_name"] = feature_name
                    best_split["split_values"] = value
                    best_split["children"] = children
                    best_split["info_gain"] = info_gain
                    max_info_gain = info_gain   
        return best_split
    


    def get_attr_type(self, dataset, feature_name): 
        pass



    def continuous_split(self, dataset, feature_index, threshold, y_parent):
        left = dataset[dataset[feature_index] <= {threshold}]
        right = dataset[dataset[feature_index] > {threshold}]
        children = [left, right]
        info_gain = self.binary_info_gain(y_parent, left.iloc[:,-1], right.iloc[:,-1])
        return children, info_gain
    


    def binary_discrete_split(self, dataset:DataFrame, feature_index:str, value:any, y_parent):
        left = dataset[dataset[feature_index] == {value}]
        right = dataset[dataset[feature_index] != {value}]
        children = [left, right]
        info_gain = self.binary_info_gain(y_parent, left.iloc[:,-1], right.iloc[:,-1])
        return children, info_gain



    def multiclass_discrete_split(self, dataset, feature_index, values, y_parent):
        labels = pd.unique(values)
        children = []
        for label in labels:
            child_dataset = dataset[dataset[feature_index] == label]
            children.append(child_dataset)
        info_gain = self.info_gain_multiclass(y_parent, children)
        return children, info_gain



    def info_gain_multiclass(self, y_parent:Series, children, y_parent_index) -> float:
        '''Get the information gain for a node'''
        children_impurity_sum = 0
        for child in children:
            children_impurity_sum += len(child) / len(y_parent) * self.get_impurity(child.iloc[:, y_parent_index])
        return self.get_impurity(y_parent) - children_impurity_sum



    def get_impurity(self, y):
        return self.gini_index(y) if self.criterium=='gini' else self.entropy(y)



    def entropy(self, y:Series) -> float:
        '''Get the entropy value for a node'''
        class_labels = pd.unique(y)
        entropy = 0
        for label in class_labels:
            p_cls = len(y[y == label]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    


    def gini_index(self, y:Series) -> float:
        '''Get the gini value for a node'''
        class_labels = pd.unique(y)
        gini = 0
        for label in class_labels:
            p_cls = len(y[y == label]) / len(y)
            gini += p_cls**2
        return 1 - gini



    def binary_info_gain(self, y_parent:Series, y_left:Series, y_right:Series) -> float:
        '''Get the information gain for a node'''
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        return self.get_impurity(y_parent) - (weight_left*self.get_impurity(y_left) + weight_right*self.get_impurity(y_right))
    


    


    





    



    


   
    


    
    def predict(self, X_test:DataFrame) -> list[any]:
        predictions = list(map(lambda row: self.make_prediction(row, self.root_node), X_test.itertuples(index=False)))
        return predictions



    def make_prediction(self, csv_row: tuple, node:DTNode, X_test:DataFrame) -> any:
        '''Predict prediction for each row in dataframe'''
        if node.leaf_value is not None: return node.leaf_value
        feature_val = csv_row.iloc[node.feature_index]
        if self.get_attr_type(feature_val) == 'multiclass':
            for value in X_test.iloc[:,node.feature_index]:
                return self.make_prediction(csv_row, node.left_node, X_test)
        else:
            if feature_val==PRIMEIRO_VALOR
                return self.make_prediction(csv_row, node.left_node, X_test)
            return self.make_prediction(csv_row, node.right_node, X_test)


    

    