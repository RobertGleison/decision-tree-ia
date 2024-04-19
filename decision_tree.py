from node import Node
import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=None) -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        Y = dataset.iloc[:,-1]
        num_samples, num_features = dataset.shape
        
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(leaf_value=leaf_value)
    

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset.iloc[:, feature_index]
            possible_thresholds = pd.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset.iloc[:, -1], dataset_left.iloc[:, -1], dataset_right.iloc[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = dataset[dataset[feature_index] <= threshold] 
        dataset_right = dataset[dataset[feature_index] > threshold] 
        return dataset_left, dataset_right
    
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    

    def entropy(self, y):
        ''' function to compute entropy '''
        class_labels = pd.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    

    def gini_index(self, y):
        ''' function to compute gini index '''
        class_labels = pd.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        Y = list(Y)
        return max(Y, key=Y.count)
    

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    

    def fit(self, X, y):
        ''' function to train the tree '''
        print("shape X: ", X.shape)
        print("shape Y: ", y.shape)
        dataset = pd.concat((X, y), axis=1, ignore_index=True)
        self.root = self.build_tree(dataset)
    

    def predict(self, X):
        ''' function to predict new dataset '''
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.leaf_value!=None: return tree.leaf_value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
