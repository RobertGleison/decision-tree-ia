import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier as DecisionTreeModel
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSKLearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, train_test_split


from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus





def main():
    df = pd.read_csv('csv_files/diabetes_dataset.csv')

    # Separate features and target variable
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
   

    # dt = DecisionTreeModel(min_samples_split=2, max_depth=4)
    dt = DecisionTreeSKLearn(min_samples_split=2, max_depth=3)


    ## escolher entre cross validation ou fazer um só predict - depth 4
    # cross_validation(dt, target, features)   ##  
    teste(dt, target, features)                ## 


#   Better Decision Tree Visualisation
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names = features.columns,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())
     
    

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
        # dt.TreePrinter()

        y_pred = dt.predict(X_test)

        print(accuracy_score(y_test, y_pred))
        


if __name__ == "__main__":
    main()