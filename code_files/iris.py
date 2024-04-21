import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut


def main():
    """Author: Sophia Cheto"""
    df = pd.read_csv('csv_files/iris.csv')
    # Map categorical values to discrete values
    class_mapping = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica' : 3
}

    df['class'] = df['class'].map(class_mapping)
    df.drop(df.columns[0], axis=1, inplace=True)

    # Separate features and target variable
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
   

    dt = DecisionTreeModel(min_samples_split=2, max_depth=2)
    # dt = DecisionTreeSKLearn(min_samples_split=2, max_depth=4)

    # Perform Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()
    accuracies = []

    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
   
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
    # print(accuracies)
    mean_accuracy = np.mean(accuracies)

    # dt.fit(X_train,y_train)

    # model_Y = dt.predict(X_test)
    print("Mean Accuracy Model:", mean_accuracy)
    # print("accuracy: ", accuracy_score(model_Y, y_test))
    # dt.TreePrinter()

if __name__ == "__main__":
    main()