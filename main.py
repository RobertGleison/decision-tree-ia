import pandas as pd
from pandas import DataFrame
from sophia.decision_tree import DecisionTree
from sophia.node import Node
from sophia.statistics import StatisticalAnalysis
import numpy as np
import pydotplus
from IPython.display import Image 
from joblib import load
import time

IRIS_CSV = 'csv_files/iris.csv'
RESTAURANT_CSV = 'csv_files/restaurant.csv'
WEATHER_CSV = 'csv_files/weather.csv'
CONNECT4_CSV = 'csv_files/connect4.csv'


def main():
    chose_csv = _print_options()
    df = pd.read_csv(chose_csv)

    
    if chose_csv == CONNECT4_CSV:
        dt = load('sophia/dt_connect4.joblib')
    else: 
        df.drop(['ID'], axis=1, inplace=True)

        start = time.time()
        dt = DecisionTree(dataset=df)
        dt.fit(df)
        end = time.time()
        print(end-start)

        StatisticalAnalysis(df)


    target = df.iloc[:,-1]
    colors = {key:value for (value, key) in zip(["#bad9d3", "#d4b4dd", "#fdd9d9"], pd.unique(target))}
    make_dot_representation(dt, colors)

    predict(dt, df)


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
    return csvs[chose_csv]


def predict(dt: DecisionTree, df: DataFrame):
    print("\n\nPREDICTION ---------")
    features = df.iloc[:,:-1] 
    features_names = features.columns
    X_test = []
    for feature in features_names:
        feature_value = input(feature + "? ")
        if feature_value.replace(".", "").isnumeric():
            X_test.append(float(feature_value))
            continue
        if feature_value.upper() == 'FALSE': 
            X_test.append("False")
            continue
        if feature_value.upper() == 'TRUE': 
            X_test.append("True")
            continue
        X_test.append(feature_value)
    
    test = pd.DataFrame([X_test], columns=features_names)
    result = dt.predict(test)[0]
    print("\nPREDICTION: ", result)

    

def make_dot_representation(dt: DecisionTree, colors: dict) -> None:
    dot_data = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\"] ;\n"
    dot_data += "edge [fontname=\"times\"] ;\n"
    dot_data += _build_dot_node(dt.root, colors)
    dot_data += "}"
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png('graph.png')
    Image(graph.create_png())



def _build_dot_node(node: Node, colors: dict) -> str:
    dot_data = ""
    if node.is_leaf:
        color = colors[node.value]
        if node.ispure:
            simbolo = '*'
        else: simbolo = ''
        dot_data += f"{id(node)} [label=\"{node.value}\", xlabel=\"{node.size}{simbolo}\", fillcolor=\"{color}\"] ;\n"
    else:
        dot_data += f"{id(node)} [label=\"{node.feature_name}?\", xlabel=\"{node.size}\"] ;\n"
        for i, child in enumerate(node.children):
            if type(node.value) == np.ndarray:
                split_value = node.value[i]
            else: split_value = node.value
            if node.split_type == 'discrete':
                dot_data += f"{id(node)} -> {id(child)} [label=\"{split_value}\"] ;\n"
            else:
                if i==0: simbolo = '<='
                else: simbolo = '>'
                dot_data += f"{id(node)} -> {id(child)} [label=\"{simbolo}{split_value}\"] ;\n"
            dot_data += _build_dot_node(child, colors)
    return dot_data


main()


