from decision_tree_classifier import DecisionTreeClassifier as DecisionTreeModel
from sklearn.model_selection import LeaveOneOut, KFold
from pandas import Series, DataFrame

class StatisticalAnalysis:
    
    def __init__(self, dataframe: DataFrame, samples: int, depth: int, criterium: str) -> None:
        self.dt = DecisionTreeModel(min_samples_split=samples, max_depth=depth, criterium=criterium)
        self.df = dataframe
        self.target_data = self.df.iloc[:,-1]
        self.features_data = self.df.iloc[:,:-1]


    def analysis(self):
        if len(self.df)<50:
            accuracies, test_size = self._leave_one_out_cross_validation()
            print(f"Cross validation type: Leave One Out")
        else:
            accuracies, test_size = self._k_fold_cross_validation()
            print(f"Cross validation type: K-Fold")
        mean_accuracy = sum(accuracies) / len(accuracies)
        self._print_statistics(mean_accuracy, test_size)

        

    def _leave_one_out_cross_validation(self) -> tuple[list, int]:
        loo = LeaveOneOut()
        accuracies = []
        for train_index, test_index in loo.split(self.features_data):
            X_train, X_test = self.features_data.iloc[train_index], self.features_data.iloc[test_index]
            y_train, y_test = self.target_data.iloc[train_index], self.target_data.iloc[test_index]
            self.dt.fit(X_train, y_train)
            y_pred = self.dt.predict(X_test)
            accuracies.append(self._accuracy_score(y_test, y_pred)) 
        return accuracies, y_test.shape[0]
    
    
    def _k_fold_cross_validation(self, n_test: int = 10) -> tuple[list, int]:
        kf = KFold(n_splits=n_test)
        accuracies = []

        for train_index, test_index in kf.split(self.features_data):
            X_train, X_test = self.features_data.iloc[train_index], self.features_data.iloc[test_index]
            y_train, y_test = self.target_data.iloc[train_index], self.target_data.iloc[test_index]

            self.dt.fit(X_train, y_train)
            y_pred = self.dt.predict(X_test)
            accuracies.append(self._accuracy_score(y_test, y_pred))
        return accuracies, y_test.shape[0]
    

    def _accuracy_score(self, y_test: Series, y_pred: Series) -> float:
        total_counter = 0
        right_predictions = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test.iloc[i]: right_predictions += 1
            total_counter += 1
        return right_predictions / total_counter
    

    def _print_statistics(self, mean_accuracy: float, test_size: int) -> None:
        print(f"Model test size: {test_size} rows")
        print(f"Model test size: {len(self.df) - test_size} rows")
        print(f"Model accuracy: {(mean_accuracy * 100):.2f}%" )