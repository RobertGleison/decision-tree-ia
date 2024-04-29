from  decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from pandas import Series, DataFrame
from IPython.display import Image  
from node import DTNode
from time import time
import pandas as pd
import numpy as np
import pydotplus
import os



IRIS_CSV = 'csv_files/iris.csv'
RESTAURANT_CSV = 'csv_files/restaurant.csv'
WEATHER_CSV = 'csv_files/weather.csv'



def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time() - start_time
        print(f"\nExecution time: {end_time:.2f} seconds")
        return result
    return wrapper



def main() -> None:
    chose_csv, samples, depth, criterium  = _print_options()
    df = pd.read_csv(chose_csv)
    df.drop(['ID'], axis=1, inplace=True)

    target = df.iloc[:,-1]
    features = df.iloc[:,:-1]
    dt_model = DecisionTreeModel(min_samples_split=samples, max_depth=depth, criterium=criterium)
   
    if chose_csv == IRIS_CSV:
        accuracies, test_size = _k_fold_cross_validation(dt_model, target, features, 10)
    if chose_csv == RESTAURANT_CSV:
        accuracies, test_size = _k_fold_cross_validation(dt_model, target, features, 5)
    if chose_csv == WEATHER_CSV:
        accuracies, test_size = _leave_one_out_cross_validation(dt_model, target, features)
    
    mean_accuracy = sum(accuracies)/len(accuracies)
    _print_statistics(mean_accuracy, test_size, features, chose_csv)



def _print_statistics(mean_accuracy: float, test_size: int, features: Series, chose_csv: str) -> None:
    print(f"Model test size: {test_size} rows")
    print(f"Model test size: {features.shape[0] - test_size} rows")
    print(f"Model accuracy: {(mean_accuracy * 100):.2f}%" )
    if chose_csv == WEATHER_CSV: print(f"Cross validation type: Leave One Out")
    else: print(f"Cross validation type: K-Fold")


# Melhorar execption handler
def _print_options() -> None:
    csvs = {1: 'csv_files/iris.csv',
            2: 'csv_files/restaurant.csv',
            3: 'csv_files/weather.csv'}
    try:
        print("Choose the dataset to train the Decision Tree:"
                "\n1 - Iris.csv\n"
                "2 - Restaurant.csv\n"
                "3 - Weather.csv\n")
        chose_csv = int(input("Dataset escolhido: "))
        samples = int(input("Escolha um número mínimo de linhas para split (recomendado: 2-5): "))
        depth = int(input("Escolha a profundidade máxima da Decision Tree (recomendado: 5-10): "))
        criterium = input("Escolha o critério de decisão de atributos ('gini' ou 'entropy'): ")
        # folds = int(input("Escolha quantidade de divisões para treino (recomendado: 1,5 ou 10): "))
        return csvs[chose_csv], samples, depth, criterium
    
    except: 
        os.system('clear')
        print("Enter a valid option for dataset")
        _print_options()


@timer
def _leave_one_out_cross_validation(dt: DecisionTreeModel, target: Series, features: DataFrame) -> tuple[list, int]:
    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(_accuracy_score(y_test, y_pred)) 
    _make_dot_representation(dt, features, target)
    return accuracies, y_test.shape[0]


@timer
def _k_fold_cross_validation(dt: DecisionTreeModel, target: Series, features: DataFrame, n_test: int =10)-> tuple[list, int]:
    kf = KFold(n_splits=n_test)
    accuracies = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(_accuracy_score(y_test, y_pred))
    _make_dot_representation(dt, features, target)
    return accuracies, y_test.shape[0]
    


def _accuracy_score(y_test: Series, y_pred: Series) -> float:
    total_counter = 0
    right_predictions = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test.iloc[i]: right_predictions += 1
        total_counter += 1
    return right_predictions / total_counter



def _make_dot_representation(dt: DataFrame, features: DataFrame, target: Series) -> None:
    dot_data = "digraph Tree {\nnode [shape=box] ;\n"
    dot_data += _build_dot_node(dt.root)
    dot_data += "}"
    png_file_path = os.path.join(os.getcwd(), 'decision_tree.png')
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png('graph.png')
    Image(graph.create_png())



def _build_dot_node(node: DTNode) -> str:
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