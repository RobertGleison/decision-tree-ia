from dt_node import DTNode
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class DecisionTreeClassifier:
    def __init__(self, max_depth:int=None, min_samples_split:int=None, criterium:str='entropy') -> None:
        self.root:DTNode = None
        self.max_depth:int = max_depth 
        self.min_samples_split:int = min_samples_split 
        self.criterium:int = criterium


    def build_tree(self, dataset:DataFrame, curr_depth:int=0) -> DTNode:
        '''Construct the Decision Tree from the root node''' 
        X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples, num_features = X.shape
        
        if num_samples>=self.min_samples_split and curr_depth<self.max_depth and not self.is_pure(y):
            best_split = self.get_best_split(dataset, num_features-1) 
            if best_split["info_gain"]>0:
                left_node = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_node = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return DTNode(best_split["feature_index"], best_split["feature_name"], best_split["threshold"], right_node, left_node, best_split["info_gain"])
            
        return DTNode(leaf_value=self.calculate_leaf_value(y))
    

    def get_best_split(self, dataset:DataFrame, num_features:int) -> dict:
        '''Get the best split for a node'''
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_name = dataset.columns[feature_index]
            feature_values = dataset.iloc[:, feature_index]
            possible_thresholds = pd.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_name, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset.iloc[:, -1], dataset_left.iloc[:, -1], dataset_right.iloc[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["feature_name"] = feature_name
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    

    def split(self, dataset:DataFrame, feature_name:str, threshold:any) -> tuple[DataFrame, DataFrame]:
        '''Get a split for a node based on a row and column'''
        dataset_left = dataset.query(f"{feature_name} <= {threshold}")
        dataset_right = dataset.query(f"{feature_name} > {threshold}")
        return dataset_left, dataset_right
    

    def is_pure(self, target_column:Series) -> bool:
        target_column = set(target_column)
        return len(target_column) == 1


    def information_gain(self, y_parent:Series, y_left:Series, y_right:Series) -> float:
        '''Get the information gain for a node'''
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        if self.criterium=="gini":
            gain = self.gini_index(y_parent) - (weight_left*self.gini_index(y_left) + weight_right*self.gini_index(y_right))
        else:
            gain = self.entropy(y_parent) - (weight_left*self.entropy(y_left) + weight_right*self.entropy(y_right))
        return gain
    

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
        

    def calculate_leaf_value(self, y:Series) -> any:
        '''Get the majority of results in a leaf node'''
        list_y = list(y)
        return max(set(list_y), key=list_y.count)
    

    def fit(self, X:DataFrame, y:Series) -> None:
        '''Fit the tree with a csv for trainning'''
        dataset = pd.concat((X, y), axis=1)
        self.root = self.build_tree(dataset)
    

    def predict(self, X_test:DataFrame) -> list[any]:
        '''Predict results based on the trained tree and a new dataframe test'''
        samples, _ = X_test.shape
        predictions = []
        for row_index in range(samples):
            csv_row = X_test.iloc[row_index]
            predictions.append(self.make_prediction(csv_row, self.root, X_test))
        return predictions
    

    def make_prediction(self, csv_row: tuple, node:DTNode, X_test:DataFrame) -> any:
        '''Predict prediction for each row in dataframe'''
        if node.leaf_value is not None: return node.leaf_value
        feature_val = csv_row.iloc[node.feature_index]
        if feature_val<=node.threshold:
            return self.make_prediction(csv_row, node.left_node, X_test)
        return self.make_prediction(csv_row, node.right_node, X_test)
    

#  PRETTY PRINTING ------------
    def height(self, node):
        if node is None: return 0
        return max(self.height(node.left_node), self.height(node.right_node)) + 1;
    
    def getcol(self, h):
        if h == 1:
            return 1
        return self.getcol(h-1) + self.getcol(h-1) +1
    
    def printTree(self, M, root, col, row, height):
        if root is None: return
        M[row][col] = root
        self.printTree(M, root.left_node, col - pow(2, height - 2), row + 1, height - 1);
        self.printTree(M, root.right_node, col + pow(2, height - 2), row + 1, height - 1);
    
 
    def TreePrinter(self):
        h = self.height(self.root);
        col = self.getcol(h);
        M = []
        for i in range(h): 
            M.append([])
            for j in range(col):
                M[i].append(0)


        self.printTree(M, self.root, col // 2, 0, h);
        for i in range (h):
            for j in range (col):
                if (M[i][j] == 0):
                    print("   ", end="")
                else:
                    if M[i][j].leaf_value is None:
                        print("N" + str(M[i][j].feature_index) + " ", end="")
                    else:
                        print("F" + str(M[i][j].leaf_value) + " ", end="")
            print()