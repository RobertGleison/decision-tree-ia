class DTNode:
    def __init__(self, feature_index=None, feature_name=None, threshold = None, children=None, info_gain=None, leaf_value=None) -> None:
        self.feature_index = feature_index    # número da coluna do atributo que analisamos
        self.feature_name = feature_name      # nome do atributo que analisamos
        self.children = children
        self.info_gain = info_gain 

        # se for binario
        self.threshold = threshold  

        ## se for folha:
        self.leaf_value = leaf_value 


