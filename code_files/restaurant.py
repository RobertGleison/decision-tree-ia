import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier as DecisionTreeModel
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

    # Tira os valores categoricos e coloca valores discretos 
    yes_no_mapping = {'Yes': 1, 'No': 0}
    none_mapping = {'None': 1, 'Some': 0, 'Full': 0}
    some_mapping = {'None': 0, 'Some': 1, 'Full': 0}
    full_mapping = {'None': 0, 'Some': 0, 'Full': 1}
    french_mapping = {'French': 1, 'Thai': 0, 'Burger': 0, 'Italian': 0}
    thai_mapping = {'French': 0, 'Thai': 1, 'Burger': 0, 'Italian': 0}
    burger_mapping = {'French': 0, 'Thai': 0, 'Burger': 1, 'Italian': 0}
    italian_mapping = {'French': 0, 'Thai': 0, 'Burger': 0, 'Italian': 1}
    price_mapping = {'$': 1, '$$': 2, '$$$': 3}
    # est = lambda str: (str.replace('>', '')).split('-')[0]
    est_mapping = {'0-10': 0, '10-30': 10, '30-60': 30, '>60': 60}

    # df['Class'] = df['Class'].map(yes_no_mapping)
    # target = df.iloc[:, -1]

    # df['Alt'] = df['Alt'].map(yes_no_mapping)
    # df['Bar'] = df['Bar'].map(yes_no_mapping)
    # df['Fri'] = df['Fri'].map(yes_no_mapping)
    # df['Hun'] = df['Hun'].map(yes_no_mapping)
    # df['Rain'] = df['Rain'].map(yes_no_mapping)
    # df['Res'] = df['Res'].map(yes_no_mapping)
    # df['Price'] = df['Price'].map(price_mapping)
    # df['Est'] = df['Est'].map(est_mapping)

    df['NonePat'] = df['Pat'].map(none_mapping)
    df['SomePat'] = df['Pat'].map(some_mapping)
    df['FullPat'] = df['Pat'].map(full_mapping)

    df['French'] = df['Type'].map(french_mapping)
    df['Thai'] = df['Type'].map(thai_mapping)
    df['Burger'] = df['Type'].map(burger_mapping)
    df['Italian'] = df['Type'].map(italian_mapping)

    df.drop(['ID'], axis=1, inplace=True)
    df.drop(['Pat', 'Type'], axis=1, inplace=True)

    # Separate features and target variable
    target = df['Class']
    df.drop(['Class'], axis=1, inplace=True)
    features = df.copy()
    # df = pd.concat([features, target], axis=1)

  
    
    dt = DecisionTreeModel(min_samples_split=2, max_depth=10)
    # dt = DecisionTreeSKLearn(min_samples_split=2, max_depth=3)

    ## escolher entre cross validation ou fazer um só predict
    # cross_validation(dt, target, features)
    teste(dt, target, features)
    


    

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
        


if __name__ == "__main__":
    main()


# Separa a última coluna do csv do resto do csv (ultima coluna = target, o resto = features)
# features = df.iloc[:,:-1]
# target = df.iloc[:,-1]


# # Separa 30% das linhas do csv pra teste da arvore dps q ela estiver treinada, e 70% pra treino da arvore. 
# # Separa os 70% de treino em 2 dataframes, um só com a ultima coluna e outra com o resto
# # Normalmente X -> Atributos que usamos pra prever y, y -> o atributo que queremos prever 
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)


# # Cria uma árvore de decisão e treina ela começando pela função fit. Passamos os dados de treino
# dt = DecisionTreeClassifier(min_samples_split=1, max_depth=10)
# dt.fit(X_train,y_train)

# # Get the results based on the train data. After that we need to compare the result with the original test data
# # model_Y = dt.predict(X_test)
# # dt.print_tree(dt.root)

# dt.TreePrinter()

