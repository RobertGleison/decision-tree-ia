class DTNode:
    def __init__(self, feature_index=None, feature_name=None, threshold = None, right_node=None, left_node=None, info_gain=None, leaf_value=None) -> None:
        self.feature_index = feature_index 
        self.feature_name = feature_name 
        self.threshold = threshold 
        self.right_node = right_node 
        self.left_node = left_node
        self.info_gain = info_gain 
        self.leaf_value = leaf_value 
