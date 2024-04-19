class Node:
    def __init__(self, feature_index=None, feature_name=None, threshold = None, right_node=None, left_node=None, info_gain=None, leaf_value=None) -> None:
        self.feature_index = feature_index # index da coluna esolhida para split
        self.feature_name = feature_name # Nome da coluna escolhida para split
        self.threshold = threshold # Valor escolhido para split da coluna escolhida
        self.right_node = right_node 
        self.left_node = left_node
        self.info_gain = info_gain # Ver o slides de elementos que mandei no grupo. Mas basicamente é o quão bom foi o split

        self.leaf_value = leaf_value # Somente para nós folhas. È o valor que queremos de todas as linhas do csv nessa folha. 

    def __str__(self) -> str:
        return f"Node(feature_index={self.feature_index}, " \
           f"feature_name={self.feature_name}, " \
           f"threshold={self.threshold}, " \
           f"right_node={self.right_node}, " \
           f"left_node={self.left_node}, " \
           f"info_gain={self.info_gain}, " \
           f"leaf_value={self.leaf_value})"



