class DTNode:
    def __init__(self, feature_index=None, feature_name=None, threshold = None, children=None, info_gain=None, leaf_value=None) -> None:
        self.feature_index = feature_index   
        self.feature_name = feature_name      
        self.children = children
        self.info_gain = info_gain 
        self.split_values = []  
        self.leaf_value = leaf_value 


