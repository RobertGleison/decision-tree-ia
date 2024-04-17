class Node:
    def __init__(self, split_attribute=None, threshold = None, right_node=None, left_node=None, leaf_value=None) -> None:
        self.split_attribute = split_attribute
        self.attribute_threshold = threshold
        self.right_node = right_node
        self.left_node = left_node

        # for leaf node
        self.leaf_value = leaf_value

