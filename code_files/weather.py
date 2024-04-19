import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier

# Lê o csv
df = pd.read_csv('csv_files/weather.csv')
 

# Tira os valores categoricos e coloca valores discretos (linha 10-28)
play_mapping = {
    'yes': 1,
    'no': 0
}
weather_mapping = {
    'sunny': 1,
    'rainy': 2,
    'overcast': 3
}

windy_mapping = {
    True: 1,
    False: 0
}

df['Windy'] = df['Windy'].map(windy_mapping)
df['Weather'] = df['Weather'].map(weather_mapping)
df['Play'] = df['Play'].map(play_mapping)

# Separa a última coluna do csv do resto do csv (ultima coluna = target, o resto = features)
features = df.iloc[:,:-1]
target = df.iloc[:,-1]

# Separa 30% das linhas do csv pra teste da arvore dps q ela estiver treinada, e 70% pra treino da arvore. 
# Separa os 70% de treino em 2 dataframes, um só com a ultima coluna e outra com o resto
# Normalmente X -> Atributos que usamos pra prever y, y -> o atributo que queremos prever 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)


# Cria uma árvore de decisão e treina ela começando pela função fit. Passamos os dados de treino
dt = DecisionTreeClassifier(min_samples_split=1, max_depth=10)
dt.fit(X_train,y_train)

# model_Y = dt.predict(X_test)
# dt.print_tree()

