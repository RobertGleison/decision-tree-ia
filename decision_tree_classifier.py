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
        self._attr_types = {}

    

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        '''Fit the tree with a DataFrame for trainning'''
        dataset = pd.concat((X_train, y_train), axis=1)
        self._map_attr_types(dataset)
        self.root = self._build_tree(dataset)



    def _build_tree(self, dataset: DataFrame, curr_depth: int = 0) -> DTNode:
        '''Construct the Decision Tree from the root node''' 
        # print(curr_depth)
        X_train, y_train = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples = X_train.shape[0]
        
        if num_samples<self.min_samples_split or curr_depth==self.max_depth or self._is_pure(y_train): # Usei um early return aq
            return DTNode(leaf_value=self._calculate_leaf_value(y_train))
        
        best_split = self._get_best_split(dataset) 
        if best_split == {}: return DTNode(leaf_value=self._calculate_leaf_value(y_train)) # Ou é uma folha ou o split dividiu 50/50. Se o melhor split dividiu 50/50, é pq todos os splits possíveis são ou folhas ou dividem 50/50. Ambos os casos CREIO EU, NA MINHA CABEÇA, melhor retornar uma folha. No caso do 50/50, retorna um valor aleatorio.

        children = []
        for child in best_split["children"]:
            children.append(self._build_tree(child, curr_depth+1))
        return DTNode(best_split["feature_index"], best_split["feature_name"], children, best_split["info_gain"], best_split["split_values"], best_split["split_type"])


    
    def _calculate_leaf_value(self, y_train: Series) -> any: # Antes só funcionava pra folhas puras, agr funciona pra folhas 50/50 caso o info gain seja 0
        '''Get value of the majority of results in a leaf node'''
        list_y = list(y_train)
        max_count = max(list_y.count(item) for item in set(list_y)) # Máximo contador da lista
        most_common_values = [item for item in set(list_y) if list_y.count(item) == max_count] # Cria lista com valores com contador máximo
        return random.choice(most_common_values)
    


    def _is_pure(self, target_column: Series) -> bool:
        '''Check if a node have only one type of target value'''
        return len(set(target_column)) == 1
    


    def _get_best_split(self, dataset: DataFrame) -> dict:
        '''Get the best split for a node'''
        best_split = {}
        max_info_gain = 0
        y_train = dataset.iloc[:, -1]
        train_columns = dataset.shape[1] - 1 

        for feature_index in range(train_columns):
            feature_name = dataset.columns[feature_index]       
            values = dataset.iloc[:, feature_index]    
            attr_type = self._get_attr_type(feature_name)

            if attr_type == 'discrete': 
                children, info_gain = self._discrete_split(dataset, feature_index, pd.unique(values), y_train)
                max_info_gain = self._update_best_split(best_split, info_gain, max_info_gain, attr_type, children, pd.unique(values.map(lambda x: str(x))), feature_name, feature_index)
                continue

            for value in pd.unique(values):
                if attr_type == 'continuous': children, info_gain = self._continuous_split(dataset, feature_index, value, y_train)
                if children is None: continue
                max_info_gain = self._update_best_split(best_split, info_gain, max_info_gain, attr_type, children, value, feature_name, feature_index)
        return best_split
    


    def _update_best_split(self, best_split: dict, info_gain: float, max_info_gain: float, split_type: str, children: list, value: any, feature_name: str, feature_index: int) -> float:
        if info_gain > max_info_gain:
            best_split["feature_index"] = feature_index
            best_split["feature_name"] = feature_name
            best_split["split_values"] = value
            best_split["split_type"] = split_type
            best_split["children"] = children
            best_split["info_gain"] = info_gain
            return info_gain
        return max_info_gain
    


    def _map_attr_types(self, dataset: DataFrame) -> None:
        for feature_name in dataset.columns:
            if type(dataset.iloc[0][feature_name]) == np.int64 or type(dataset.iloc[0][feature_name]) == np.float64: 
                self._attr_types[feature_name] = 'continuous'
                continue
            self._attr_types[feature_name] = 'discrete'



    def _get_attr_type(self, feature_name: str) -> str: 
        return self._attr_types[feature_name]



    def _continuous_split(self, dataset: DataFrame, feature_index: int, threshold: any, y_parent: Series) -> tuple[list, float]:
        '''Split the DataFrame with a continuous value'''
        left = dataset[dataset.iloc[:, feature_index] <= threshold]
        right = dataset[dataset.iloc[:, feature_index] > threshold]
        children = [left, right]
        info_gain = self._info_gain(dataset, children, y_parent)
        return children, info_gain
    


    def _discrete_split(self, dataset: DataFrame, feature_index: int, values: Series, y_parent: Series) -> tuple[list, float]:
        '''Split the DataFrame with a discrete and multiclass value'''
        labels = list(pd.unique(values))
        children = []
        for label in labels:
            child_dataset = dataset[dataset.iloc[:, feature_index] == label]
            children.append(child_dataset)
        info_gain = self._info_gain(dataset, children, y_parent)
        return children, info_gain
    


    def _info_gain(self, parent_dataset: DataFrame, children: list, y_parent: Series) -> float:
        '''Get the information gain for a node splitted by a multiclass discrete value'''
        children_impurity_sum = 0
        for child_dataset in children:
            children_impurity_sum += child_dataset.shape[0] / parent_dataset.shape[0] * self._get_impurity(child_dataset.iloc[:, -1])
        return self._get_impurity(y_parent) - children_impurity_sum



    def _get_impurity(self, y_train: Series) -> float:
        '''Get the impurity of the node'''
        return self._gini_index(y_train) if self.criterium=='gini' else self._entropy(y_train)



    def _entropy(self, y_train: Series) -> float:
        '''Get the entropy value for a node'''
        class_labels = pd.unique(y_train)
        entropy = 0
        for label in class_labels:
            label_positives = len(y_train[y_train == label]) / len(y_train)
            entropy += -(label_positives * np.log2(label_positives))
        return entropy
    


    def _gini_index(self, y_train: Series) -> float:
        '''Get the gini value for a node'''
        class_labels = pd.unique(y_train)
        gini = 0
        for label in class_labels:
            label_positives = len(y_train[y_train == label]) / len(y_train)
            gini += label_positives**2
        return 1 - gini



    def predict(self, X_test: DataFrame) -> list:
        '''Predict target column for a dataframe'''
        return [self.make_prediction(row, self.root, X_test) for _, row in X_test.iterrows()]



    def make_prediction(self, row: tuple, node: DTNode, X_test: DataFrame) -> Union[any, None]:
        '''Predict target for each row in dataframe'''
        if node.leaf_value is not None: 
            return node.leaf_value
        
        value = row.iloc[node.feature_index] 
        attribute = node.feature_name

        if node.split_type == 'discrete': 
            for i, node_value in enumerate(node.split_values):
                if value == node_value:
                    return self.make_prediction(row, node.children[i], X_test)  
        
        elif node.split_type == 'continuous':
            if value <= node.split_values:
                return self.make_prediction(row, node.children[0], X_test)  
            return self.make_prediction(row, node.children[1], X_test)

        return None

    

    