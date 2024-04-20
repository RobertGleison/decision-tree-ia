from node import Node
import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=None, criterium='entropy') -> None:
        self.root = None
        self.max_depth = max_depth # tamanho máximo da arvore
        self.min_samples_split = min_samples_split # Quantidade mínima de linhas do csv dentro de uma folha. Eu expliquei o porque no vídeo q fiz
        self.criterium = criterium


    # Cria os nós da arvore caso ainda precise splitar, no fim retorna o nó root já com seus filhos
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
        num_samples, num_features = X.shape
        
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth and not self.is_pure(y):
            best_split = self.get_best_split(dataset, num_features-1) # Escolhe o atributo para splitar o nó
            if best_split["info_gain"]>0:
                left_node = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_node = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["feature_name"], best_split["threshold"], 
                            left_node, right_node, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(leaf_value=leaf_value)
    
    #Faz todos os splits possiveis entre colunas e valores das linhas, retorna o que te dá mais ganho. Assim escolhemos o atributo pro split de um nó
    def get_best_split(self, dataset, num_features):
        ''' function to find the best split '''
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_name = dataset.columns[feature_index]
            # if feature_name in used_attributes: continue
            feature_values = dataset.iloc[:, feature_index]
            possible_thresholds = pd.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, feature_name, threshold)
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
        # if best_split: used_attributes.add(best_split["feature_name"]) 
        return best_split
    

    def split(self, dataset, feature_index, feature_name, threshold):
        ''' function to split the data '''
        dataset_left = dataset[dataset[feature_name] <= threshold] 
        dataset_right = dataset[dataset[feature_name] > threshold] 
        return dataset_left, dataset_right
    
    def is_pure(self, target_column):
        target_column = set(target_column)
        return len(target_column) == 1


    # ver slides de elementos
    def information_gain(self, y_parent, y_left, y_right):
        ''' function to compute information gain '''
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        if self.criterium=="gini":
            gain = self.gini_index(y_parent) - (weight_left*self.gini_index(y_left) + weight_right*self.gini_index(y_right))
        else:
            gain = self.entropy(y_parent) - (weight_left*self.entropy(y_left) + weight_right*self.entropy(y_right))
        return gain
    
    # ver slides de elementos
    # entropia = (total_atual/total_pai) * gini
    def entropy(self, y):
        ''' function to compute entropy '''
        target_column = pd.unique(y)
        entropy = 0
        for attribute in target_column:
            p_cls = len(y[y == attribute]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    # gini = (negativo/total)**2 / (positivo/total)**2
    # ver slides de elementos
    def gini_index(self, y):
        ''' function to compute gini index '''
        class_labels = pd.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    # Calcula basicamente a moda de uma maneira chique. Se o nó folha for impuro (com resultados diferentes), todos eles recebem o valor do resultado que mais se repete.
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        list_Y = list(Y)
        return max(set(list_Y), key=list_Y.count)
    
    # treina a arvpre
    def fit(self, X, y):
        ''' function to train the tree '''
        dataset = pd.concat((X, y), axis=1)
        self.root = self.build_tree(dataset)
    
    # Deveria retornar uma coluna nova com predições, mas n funciona ainda
    # def predict(self, X):
    #     ''' function to predict new dataset '''
    #     preditions = [self.make_prediction(x, self.root) for x in X]
    #     return preditions
    

    # def make_prediction(self, x, node):
    #     ''' function to predict a single data point '''
    #     if node.leaf_value!=None: return node.leaf_value
    #     feature_val = x[node.feature_index]
    #     if feature_val<=node.threshold:
    #         return self.make_prediction(x, node.left)
    #     else:
    #         return self.make_prediction(x, node.right)
    
    # def print_tree(self, node=None, splits=None):
    #     ''' function to print the tree '''
    #     if node is None: node = self.root
    #     if splits is None: splits = []
    #     splits.append(node)
    #     if node.right_node is not None: splits = self.print_tree(node.right_node, splits)
    #     if node.leaf_value is not None: splits = self.print_tree(node.left_node, splits)
    #     if node.right_node is None and node.left_node is None: return splits
    #     return splits

        # else:
        #     print("X_"+str(node.feature_index), "<=", node.threshold, "?", node.info_gain)
        #     print("%sleft:" % (indent), end="")
        #     self.print_tree(node.left_node, indent + indent)
        #     print("%sright:" % (indent), end="")
        #     self.print_tree(node.right_node, indent + indent)
    
    # def print_tree(self, node=None, splits=[]):
    #     if node is None: return
    #     if node.leaf_value is not None: 
    #             print("folha: " + str(node.leaf_value))
    #             return
    #     # splits.append(node)
    #     print(str(node.feature_name)+": X_"+str(node.feature_index), "<=", node.threshold, "?", node.info_gain)
    #     self.print_tree(node.left_node, splits)
    #     self.print_tree(node.right_node, splits)