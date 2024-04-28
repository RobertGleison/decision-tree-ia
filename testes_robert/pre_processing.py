import pandas as pd
import numpy as np
import os
from  decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from IPython.display import display
from graphviz import Source
import pydotplus
from IPython.display import Image  


def main():
    # csv_name = 'csv_files/iris.csv'
    # csv_name = 'csv_files/weather.csv'
    csv_name = 'csv_files/restaurant.csv'

    df = pd.read_csv(csv_name)
    df.drop(['ID'], axis=1, inplace=True)

    # Separate features and target variable
    target = df.iloc[:,-1]
    features = df.iloc[:,:-1]
    dt_model = DecisionTreeModel(min_samples_split=2, max_depth=5)
   
    if csv_name == 'csv_files/iris.csv':
        accuracies = k_fold_cross_validation(dt_model, target, features, 10)
    if csv_name == 'csv_files/restaurant.csv':
        accuracies = k_fold_cross_validation(dt_model, target, features, 3)
    if csv_name == 'csv_files/weather.csv':
        accuracies = leave_one_out_cross_validation(dt_model, target, features)

    
    mean_accuracy = sum(accuracies)/len(accuracies)
    print(f"Mean Accuracy:", mean_accuracy)

    
    
def leave_one_out_cross_validation(dt, target, features):
    loo = LeaveOneOut()
    accuracies = []
    predictions = []
    count = 0
    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        dt.fit(X_train, y_train)
        count+=1
        y_pred = dt.predict(X_test)
        predictions.append(y_pred)
        accuracies.append(accuracy_score(y_test, y_pred)) 
        make_dot_representation(dt, features, target)
    return accuracies



def k_fold_cross_validation(dt, target, features, n_test):
    kf = KFold(n_splits=n_test)
    accuracies = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return accuracies
    


def accuracy_score(y_test, y_pred):
    total_counter = 0
    right_predictions = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test.iloc[i]: right_predictions += 1
        total_counter += 1
    return right_predictions / total_counter



def make_dot_representation(dt, features, target) -> None:
    dot_data = "digraph Tree {\nnode [shape=box] ;\n"
    dot_data += _build_dot_node(dt.root)
    dot_data += "}"
    print("Current working directory:", os.getcwd())
    png_file_path = os.path.join(os.getcwd(), 'decision_tree.png')
    print("PNG file path:", png_file_path)
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png('graph.png')
    Image(graph.create_png())

def _build_dot_node(node) -> str:
    dot_data = ""
    if node.leaf_value is not None:
        dot_data += f"{id(node)} [label=\"{node.leaf_value}\"] ;\n"
    else:
        dot_data += f"{id(node)} [label=\"{node.feature_name}\"] ;\n"
        for i, child in enumerate(node.children):
            if type(node.split_values) == np.ndarray:
                split_value = node.split_values[i]
            else: split_value = node.split_values
            dot_data += f"{id(node)} -> {id(child)} [label=\"{split_value}\"] ;\n"
            dot_data += _build_dot_node(child)
    return dot_data



if __name__ == "__main__":
    main()