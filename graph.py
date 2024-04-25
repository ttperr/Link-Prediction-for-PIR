
class Graph(object):
    
    def __init__(self) -> None:
        """initialize the graph, empty"""
        self.links = [{}, {}]# should use some sparse matrix representation instead
        
    def add_link(self, node1, node2):
        if node1 in self.links[0]:
            self.links[0][node1].add(node2)
        else:
            self.links[0][node1] = {node2}
        if node2 in self.links[1]:
            self.links[1][node2].add(node1)
        else:
            self.links[1][node2] = {node1}
    
    def neighbors(self, node, node_type) -> set:
        if not node_type in [1,2] or not node in self.links[node_type - 1]:
            return {}
        return self.links[node_type - 1][node]

    def common_neighbors(self, node1, node2):
        return len(set(self.neighbors(node1, 1)).union(set(self.neighbors(node2, 2))))
