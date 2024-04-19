import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier

# read iris.csv
df = pd.read_csv('iris.csv')

# Transform the categorical attribute in iris.csv into a numerical attribute
class_mapping = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica' : 3
}

# Map the categorial values into numerical values in 'class' columns
df['class'] = df['class'].map(class_mapping)



# Separate the dataset into train data and test data. We can use scikit learn into pre-process step.
# In practice, what this code do is: separate 30% of the rows in dataset to evaluate the result, and 70% do train
# Later we need to implement cross-validation
train_data = df.drop(columns='class')
target_data = df['class']
X_train, X_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.3, random_state=0)


# Create DecisionTreeClassifier, train the decision tree with de data for trainning
dt = DecisionTreeClassifier(min_samples_split=1, max_depth=10)
dt.fit(X_train,y_train)

# Get the results based on the train data. After that we need to compare the result with the original test data
model_Y = dt.predict(X_test)
print()

