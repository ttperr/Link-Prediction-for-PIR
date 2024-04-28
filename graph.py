
import math
import random

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
    
    def neighbors(self, node) -> set:
        for i in range(2):
            if node in self.links[i]:
                return self.links[i][node]
        return {}

    def degree(self, node):
        return len(self.neighbors(node))

    def common_neighbors(self, node1, node2):
        """
        Basic link prediction metric calculated on the user-document graph
        paper: https://onlinelibrary.wiley.com/doi/10.1002/asi.20591
        """
        res = 0
        neighbors1 = self.neighbors(node1)
        for node in neighbors1:
            res += self.rec(node2, node) / len(neighbors1)
        neighbors1 = self.neighbors(node1)
        for node in neighbors1:
            res += self.rec(node2, node) / len(neighbors1)
        return res

    def rec(self, node, node_bis):
        return len(set(self.neighbors(node)).intersection(set(self.neighbors(node_bis))))

    def neighborhood_intersection(self, node1, node2):
        """
        Return the 'intersection' of nodes neighborhood.
        This intersection is a bit skewed as the graph is bipartite.
        """
        neighbors_type1 = self.neighbors(node2)
        neighbors_type2 = self.neighbors(node1)
        intersection_type1 = set()
        intersection_type2 = set()
        for node in neighbors_type1:
            intersection_type2 = intersection_type2.union(set(self.neighbors(node)))
        intersection_type2 = intersection_type2.intersection(set(neighbors_type2))
        for node in neighbors_type2:
            intersection_type1 = intersection_type1.union(set(self.neighbors(node)))
        intersection_type1 = intersection_type1.intersection(set(neighbors_type1))
        intersection = intersection_type1.union(intersection_type2)
        return intersection

    def adamic_adar(self, node1, node2):
        """
        Adamic-Adar similarity metrics defined in https://www.sciencedirect.com/science/article/pii/S0378873303000091?via%3Dihub
        TODO: take into account node degree to make hubs less impactful
        """
        intersection = self.neighborhood_intersection(node1, node2)
        res = 0
        for node in intersection:
            res += 1/math.log(self.degree(node))
        return res

    def random_step(self, starting_node):
        neighbors = self.neighbors(starting_node)
        if len(neighbors) == 0:
            return None
        return random.choice(list(neighbors))
    
    def rooted_page_rank(self, root_node, target_node):
        bored = 0.05
        nb_steps = 50000
        hits = 0
        node = root_node
        for _ in range(nb_steps):
            if random.random() < bored:
                node = root_node
            else:
                node = self.random_step(node)
                if node == None:
                    node == root_node
                    continue
                if node == target_node:
                    hits+=1
        return hits/nb_steps
    
    