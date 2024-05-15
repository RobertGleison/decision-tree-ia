from pandas import DataFrame
from scipy import stats as st

class Node:

# dataset, children, best_split["info_gain"], best_split["feature_name"], best_split["split_type"])
    def __init__(self, dataset, children=None, value=None, info_gain=None, feature_name=None, split_type=None, is_leaf=False) -> None:
        self.dataset = dataset                              # linhas do dataset que estão no nó
        self.size = len(dataset)                            # qtd de linhas
        self.ispure = len(set(dataset.iloc[:,-1]))==1


        # FOLHA
        self.is_leaf = is_leaf                              
        self.value = value          # se for folha, valor da classe
                                    # se for gerado por feature contínua = threshold
                                    # se discreto = valor da feature 


        # NÓ 
        self.feature_name = feature_name                    # atributo usado na divisão dos filhos
        self.split_type = split_type                        # divisão dos filhos "discrete" ou "continuous"
        self.children = children                            # nós-filhos do nó atual
        self.info_gain = info_gain                          # info gain da divisão entre os filhos





    def calculate_leaf_value(self, targets: DataFrame):
        '''Get value of the majority of results in a leaf node'''
        target = list(targets)
        uniques = set(target)
        counting = [(target.count(item), item) for item in uniques]
        max_value = max(counting)[1]
        return max_value
    