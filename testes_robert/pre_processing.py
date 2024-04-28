import pandas as pd
import numpy as np
from  decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def main():
    # csv_name = 'csv_files/iris.csv'
    csv_name = 'csv_files/weather.csv'
    # csv_name = 'csv_files/restaurant.csv'

    df = pd.read_csv(csv_name)
    df.drop(['ID'], axis=1, inplace=True)

    # Separate features and target variable
    target = df.iloc[:,-1]
    features = df.iloc[:,:-1]
    dt_model = DecisionTreeModel(min_samples_split=2, max_depth=7)
   
    if csv_name == 'csv_files/iris.csv':
        accuracies = k_fold_cross_validation(dt_model, target, features)
    accuracies = leave_one_out_cross_validation(dt_model, target, features)

    mean_accuracy = sum(accuracies)/len(accuracies)
    print(f"Mean Accuracy:", mean_accuracy)

    
def leave_one_out_cross_validation(dt, target, features):
    loo = LeaveOneOut()
    accuracies = []

    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return accuracies


def k_fold_cross_validation(dt, target, features):
    kf = KFold(n_splits=10)
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


if __name__ == "__main__":
    main()