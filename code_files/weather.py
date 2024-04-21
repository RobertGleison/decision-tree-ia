import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, train_test_split


def main():
    df = pd.read_csv('csv_files/weather.csv')
    # @Author: Sophia Cheto
    # Map categorical values to discrete values
    play_mapping = {'yes': 1, 'no': 0}
    sunny_mapping = {'sunny': 1, 'rainy': 0, 'overcast': 0}
    rainy_mapping = {'sunny': 0, 'rainy': 1, 'overcast': 0}
    overcast_mapping = {'sunny': 0, 'rainy': 0, 'overcast': 1}
    windy_mapping = {True: 1, False: 0}

    df['Windy'] = df['Windy'].map(windy_mapping)
    df['Play'] = df['Play'].map(play_mapping)
    df['Rainy'] = df['Weather'].map(rainy_mapping)
    df['Sunny'] = df['Weather'].map(sunny_mapping)
    df['Overcast'] = df['Weather'].map(overcast_mapping)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)

    # Separate features and target variable
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
   

    dt = DecisionTreeModel(min_samples_split=2, max_depth=4)
    # dt = DecisionTreeSKLearn(min_samples_split=2, max_depth=4, criterion='gini')


    ## escolher entre cross validation ou fazer um só predict - depth 4
    cross_validation(dt, target, features)   ##  0,714285
    # teste(dt, target, features)                ## 0,4 - 0,6 - 0,6 - 0,4 - 0,6
    


    

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
        dt.TreePrinter()
        print()

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
        # print(y_test)
        # print(model_Y)

        # correct = 0
        # total = 0
        # for i in range(len(model_Y)):
        #     total += 1
        #     if model_Y[i] == y_test[i]: correct += 1

        print(accuracy_score(y_test, y_pred))
        


    # # Perform Leave-One-Out Cross-Validation (LOOCV)
    # loo = LeaveOneOut()
    # accuracies = []

    # for train_index, test_index in loo.split(features):
    #     X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    #     y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
    #     dt.fit(X_train, y_train)
    #     y_pred = dt.predict(X_test)
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     # dt.TreePrinter()

    # print(accuracies)
    # mean_accuracy = np.mean(accuracies)
    # print("Mean Accuracy Model:", mean_accuracy)


if __name__ == "__main__":
    main()