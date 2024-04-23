import pandas as pd
import numpy as np
from not_binary_decision_tree import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split


def main():
    # Lê o csv
    df = pd.read_csv('csv_files/restaurant.csv')

    df.Alt.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Bar.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Fri.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Hun.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Rain.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Res.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Price.replace({'$': 1, '$$': 2, '$$$': 3}, inplace = True)
    df.Est.replace({'0-10': 0, '10-30': 10, '30-60': 30, '>60': 60}, inplace = True)
    df.Class.replace({'Yes': 1, 'No': 0}, inplace = True)
    df.Pat.replace({'None': 0, 'Some': 1, 'Full': 2}, inplace = True)
    df.Type.replace({'French': 0, 'Thai': 1, 'Burger': 2, 'Italian': 3}, inplace = True)

    df.drop(['ID', 'Pat', 'Type'], axis=1, inplace=True)
    # Separate features and target variable
    target = df['Class']
    df.drop(['Class'], axis=1, inplace=True)
    features = df.copy()
  
    
    dt = DecisionTreeModel(min_samples_split=2, max_depth=10)
    # dt = DecisionTreeSKLearn(min_samples_split=2, max_depth=3)

    ## escolher entre cross validation ou fazer um só predict
    # cross_validation(dt, target, features)
    teste(dt, target, features)   
    cross_validation(dt, target, features)

    
def cross_validation(dt, target, features):
    # Perform Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()
    accuracies = []

    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        # dt.TreePrinter()
        # print()

    print(accuracies)
    mean_accuracy = np.mean(accuracies)
    print("Mean Accuracy Model:", mean_accuracy)
    return mean_accuracy



def teste(dt, target, features):
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=i)
        dt.fit(X_train, y_train)
        dt.TreePrinter()

        y_pred = dt.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print("-----------------------------------------------------------")
    


if __name__ == "__main__":
    main()