from sophia.node import Node
from pandas import DataFrame
import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self, dataset: DataFrame, max_depth: int = 100, min_samples: int = 0) -> None:
        def fill_types(dataset: pd.DataFrame):
            def mapping(tuplo):
                if tuplo[1] in [np.int64, np.float64]: return (tuplo[0], 'continuous')
                else: return (tuplo[0],'discrete')
            return dict(map(mapping, zip(dataset.columns, dataset.dtypes)))
        
        self.root: Node = None
        self.dataset: DataFrame = dataset
        self.max_depth: int = max_depth 
        self.min_samples: int = min_samples
        self.feature_types = fill_types(dataset)

    def __str__(self) -> str:
        return self.toString(self.root, "")

    def fit(self, dataset: DataFrame):
        '''Fit the tree with a trainning DataFrame'''
        self.root = self.build_tree(dataset)

    def build_tree(self, dataset: DataFrame, depth: int = 0) -> Node:
        '''Build the Decision Tree from the root node'''
        def calculate_leaf_value(targets: DataFrame):
            '''Get value of the majority of results in a leaf node'''
            target = list(targets)
            uniques = set(target)
            counting = [(target.count(item), item) for item in uniques]
            max_value = max(counting)[1]
            return max_value
        
        # print(depth)
        
        features = dataset.iloc[:,:-1]
        targets = dataset.iloc[:,-1]
        num_samples = len(dataset)

        # reaches the limit of the tree
        ispure = (len(set(targets)) == 1)
        if num_samples < self.min_samples or depth == self.max_depth or ispure or len(features.columns)==0 or num_samples==0:   ########## mudei
            return Node(value=calculate_leaf_value(targets), is_leaf=True, dataset=dataset)
        
        best_split = self.get_best_split(dataset)

        if best_split == {}: 
            return Node(value=calculate_leaf_value(targets), is_leaf=True, dataset=dataset)

        children = []
        for child in best_split["children"]:
            if len(child) == 0: 
                children.append(Node(value=calculate_leaf_value(targets), is_leaf=True, dataset=child))
            else: 
                children.append(self.build_tree(child, depth+1))
        return Node(dataset, children, best_split["value"], best_split["info_gain"], best_split["feature_name"], best_split["split_type"])
    

    def get_best_split(self, dataset: DataFrame):
        '''Get the best split for a node'''
        best_split = {}
        max_infogain = 0
        features = dataset.iloc[:,:-1]
        targets = dataset.iloc[:, -1]

        for feature_name in features.columns:
            values = self.dataset[feature_name]
            feature_type = self.feature_types[feature_name]
            parent_entropy = self.entropy(targets) 

            if feature_type == 'discrete':
                children, info_gain = self.discrete_split(dataset, feature_name, pd.unique(values), targets, parent_entropy)
                max_infogain = self.update_best_split(best_split, info_gain, max_infogain, feature_type, children, pd.unique(values.map(lambda x: str(x))), feature_name)
                continue

            for value in pd.unique(values):
                children, info_gain = self.continuous_split(dataset, feature_name, value, targets, parent_entropy)
                max_infogain = self.update_best_split(best_split, info_gain, max_infogain, feature_type, children, value, feature_name)

        return best_split
    
    def update_best_split(self, best_split: dict, info_gain: float, max_infogain: float, feature_type: str, children: list, value: any, feature_name: str):
        if info_gain > max_infogain:
            best_split["feature_name"] = feature_name
            best_split["value"] = value
            best_split["split_type"] = feature_type
            best_split["children"] = children
            best_split["info_gain"] = info_gain
            return info_gain
        return max_infogain
    
    def continuous_split(self, dataset: DataFrame, feature_name, threshold, targets, parent_entropy):
        left = dataset[dataset[feature_name] <= threshold].copy()
        right = dataset[dataset[feature_name] > threshold].copy()
        left.drop([feature_name], axis=1, inplace=True)
        right.drop([feature_name], axis=1, inplace=True)
        children = [left, right]
        info_gain = self.info_gain(dataset, children, targets, parent_entropy)
        return children, info_gain
    
    def discrete_split(self, dataset: DataFrame, feature_name, values, targets, parent_entropy):
        labels = list(values)
        children = []
        for label in labels: 
            child_dataset = dataset[dataset[feature_name] == label].copy()
            child_dataset.drop([feature_name], axis=1, inplace=True)
            children.append(child_dataset)
        info_gain = self.info_gain(dataset, children, targets, parent_entropy)
        return children, info_gain
    
    def info_gain(self, parent_dataset: DataFrame, children: list[DataFrame], parent_targets: DataFrame, parent_entropy):
        parent_length = len(parent_dataset)
        children_entropy = np.sum([(len(child_dataset) / parent_length) * self.entropy(child_dataset.iloc[:, -1]) for child_dataset in children])
        return parent_entropy - children_entropy


    def entropy(self, targets) -> float:
        '''Get the entropy value for a node'''
        counts = targets.value_counts()
        probs = counts/len(targets)
        return np.sum(-probs * np.log2(probs))
    

    def predict(self, X_test: DataFrame) -> list:
        '''Predict target column for a dataframe'''
        return [self.make_prediction(row, self.root) for _, row in X_test.iterrows()]


    def make_prediction(self, row: tuple, node: Node):
        '''Predict target for each row in dataframe'''
        if node.is_leaf: 
            return node.value
        
        value = row[node.feature_name]

        if node.split_type == 'discrete': 
            for i, node_value in enumerate(node.value):
                if value == node_value:
                    return self.make_prediction(row, node.children[i])  
        
        elif node.split_type == 'continuous':
            if value <= node.value:
                return self.make_prediction(row, node.children[0])  
            return self.make_prediction(row, node.children[1])

        return None


    def toString(self, node: Node, indent: str):
        string = ""
        if not node.children:
            return string
        
        add = " " * 5
        string += indent + f"<{node.feature_name}>" + "\n"
        indent += add

        for i in range(len(node.children)):
            child = node.children[i]

            if type(node.value) in [np.int64, np.float64]:
                if i==0: simbolo="<="
                else: simbolo=">"
                if child.is_leaf: #se for folha
                    string += indent + f"value {simbolo} {node.value}: {child.value} ({child.size})" + "\n"
                else:
                    string += indent + f"value {simbolo} {node.value}:" + "\n"
                    string += self.toString(child, indent+add)
            
            else: 
                if child.is_leaf: #se for folha
                    string += indent + f"{node.value[i]}: {child.value} ({child.size})" + "\n"  # valor da feature: valor da folha (qtd de linhas)
                else:
                    string += indent + f"{node.value[i]}:" + "\n"
                    string += self.toString(child, indent+add)
        
        return string

       