from  delete.decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from delete.statistical_analysis import StatisticalAnalysis, tree_output
from delete.data_tree import DataTree
from IPython.display import Image  
from delete.node import DTNode
from time import time
import pandas as pd
import numpy as np
import pydotplus
# import os
from joblib import load

IRIS_CSV = 'csv_files/iris.csv'
RESTAURANT_CSV = 'csv_files/restaurant.csv'
WEATHER_CSV = 'csv_files/weather.csv'
CONNECT4_CSV = 'csv_files/connect4.csv'



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

    if chose_csv == CONNECT4_CSV:
        dt_final = load('connect4_dt.joblib')

    else:
        df.drop(['ID'], axis=1, inplace=True)

        # STATISTICAL ANALYSIS
        dt_analysis = StatisticalAnalysis(df, samples, depth, criterium)
        dt_analysis.analysis()

        # TREE FOR PREDICTIONS
        dt_final = DataTree(df=df, min_samples_split=samples, max_depth=depth, criterium=criterium)
        dt_final.fit()
        dt_final.decisiontree.fill_leaf_counters(df)

    # MAKE THE GRAPH IMAGE
    target = df.iloc[:,-1]
    colors = {key:value for (value, key) in zip(["#bad9d3", "#d4b4dd", "#fdd9d9"], pd.unique(target))}
    _make_dot_representation(dt_final.decisiontree, colors)
    print(tree_output(dt_final.decisiontree.root))
    # MAKE A PREDICTION
    prediction = input("\nDo you wanna make a prediction? (y/n) ")
    while (prediction == 'y'):
        dt_final.predict()
        prediction = input("\nDo you wanna make a prediction? (y/n) ")


# Melhorar execption handler
def _print_options() -> None:
    csvs = {1: 'csv_files/iris.csv',
            2: 'csv_files/restaurant.csv',
            3: 'csv_files/weather.csv',
            4: 'csv_files/connect4.csv'}
    
    print("Choose the dataset to train the Decision Tree:"
            "\n1 - Iris.csv\n"
            "2 - Restaurant.csv\n"
            "3 - Weather.csv\n"
            "4 - Connect4.csv\n")
    chose_csv = int(input("Dataset escolhido: "))
    if chose_csv==4: return csvs[chose_csv], 0, 0, 0
    samples = int(input("Escolha um número mínimo de linhas para split (recomendado: 2-5): "))
    depth = int(input("Escolha a profundidade máxima da Decision Tree (recomendado: 5-10): "))
    criterium = input("Escolha o critério de decisão de atributos ('gini' ou 'entropy'): ")
    return csvs[chose_csv], samples, depth, criterium
    
    # except: 
    #     os.system('clear')
    #     print("Enter a valid option for dataset")
    #     _print_options()



def _make_dot_representation(dt: DecisionTreeModel, colors: dict) -> None:
    dot_data = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\"] ;\n"
    dot_data += "edge [fontname=\"times\"] ;\n"
    dot_data += _build_dot_node(dt.root, colors)
    dot_data += "}"
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png('graph.png')
    Image(graph.create_png())



def _build_dot_node(node: DTNode, colors: dict) -> str:
    dot_data = ""
    if node.leaf_value is not None:
        color = colors[node.leaf_value]
        dot_data += f"{id(node)} [label=\"{node.leaf_value}\", fillcolor=\"{color}\"] ;\n"
    else:
        dot_data += f"{id(node)} [label=\"{node.feature_name}?\"] ;\n"
        for i, child in enumerate(node.children):
            if type(node.split_values) == np.ndarray:
                split_value = node.split_values[i]
            else: split_value = node.split_values
            dot_data += f"{id(node)} -> {id(child)} [label=\"{split_value}\"] ;\n"
            dot_data += _build_dot_node(child, colors)
    return dot_data



if __name__ == "__main__":
    main()