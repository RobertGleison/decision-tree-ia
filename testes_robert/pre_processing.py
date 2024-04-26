import pandas as pd
import numpy as np
from  decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split


def main():
    
    try:
        csv_path = str(input("Enter the path of the chosen CSV file: "))
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("File not found. Please make sure you enter the correct file path.")
    except Exception as e:
        print("An error occurred:", e)


    # df = pd.read_csv('csv_files/restaurant.csv')
    df.drop(['ID'], axis=1, inplace=True)

    # Separate features and target variable
    target = df['Class']
    df.drop(['Class'], axis=1, inplace=True)
    features = df
  
    
    dt_model = DecisionTreeModel(min_samples_split=2, max_depth=4)
    dt_scikit = DecisionTreeSKLearn(min_samples_split=2, max_depth=4)

    ## escolher entre cross validation ou fazer um só predict
    # cross_validation(dt, target, features) 
    cross_validation(dt_model, target, features, 'dt_model')
    cross_validation(dt_scikit, target, features, 'dt_scikit')

    
def cross_validation(dt, target, features, model):
    # Perform Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()
    accuracies = []

    for train_index, test_index in loo.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
       

    print(accuracies)
    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy {model}:", mean_accuracy)


   


if __name__ == "__main__":
    main()