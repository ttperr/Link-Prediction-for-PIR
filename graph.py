#! /usr/bin/env python3

import csv
from datetime import datetime
import math
import random
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

EPSILON = 0.001


class Graph(object):

    def __init__(self, index1, index2, validation, dataset="AOL", name=None) -> None:
        """initialize the graph, empty"""
        self.name = str(datetime.timestamp) if name == None else name
        self.nx_graph = nx.Graph()
        self.populate_graph_from_logs(index1, index2, validation, dataset)

    def add_link(self, node1, node2):
        if self.nx_graph.has_edge(node1, node2):
            self.nx_graph[node1][node2]["weight"] += 1
        else:
            self.nx_graph.add_edge(node1, node2, weight=1)

    def populate_graph_from_logs(self, index1, index2, validation, dataset="AOL"):
        if dataset == "AOL":
            doc = "datasets/AOL4PS/data.csv"
            if validation:
                doc = "datasets/AOL4PS/training_data.csv"
            with open(doc) as f:
                reader = csv.reader(f, delimiter="\t")
                firstRow = True
                pbar = tqdm(reader, desc="Parsing queries", unit="rows")
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    self.add_link(row[index1], row[index2])
        else:
            raise NotImplementedError("Unknown dataset")

    def get_weight(self, node1, node2):
        if self.nx_graph.has_edge(node1, node2):
            return self.nx_graph[node1][node2]["weight"]
        return 0

    def neighbors(self, node) -> set:
        return set(self.nx_graph.neighbors(node))

    def degree(self, node):
        return self.nx_graph.degree[node]

    def shortest_distance(self, start, targets):
        if not self.nx_graph.has_node(start):
            return np.zeros(shape=(len(targets)))
        lengths = nx.shortest_path_length(
            self.nx_graph, start, weight=lambda u, v, e: 1 / e.get("weight", 0)
        )
        return np.array([lengths.get(arrival, 0) + EPSILON for arrival in targets])

    def weighted_shortest_distance(self, start, targets):
        """Each edge is weighted by the degree of arrival node. Going to a more popular node is less significant"""
        if not self.nx_graph.has_node(start):
            return np.zeros(shape=(len(targets)))
        lengths = nx.shortest_path_length(
            self.nx_graph,
            start,
            weight=lambda u, v, e: self.nx_graph.degree(v) / e.get("weight", 0),
        )
        return np.array([lengths.get(arrival, 0) + EPSILON for arrival in targets])

    def common_neighbors_metric(self, node1, targets):
        """Wrapper for the common neighbors metric computation meant to be run on an array of target nodes."""
        return np.array(
            [self.common_neighbors_single(node1, node2) for node2 in targets]
        )

    def common_neighbors_single(self, node1, node2):
        """
        Basic link prediction metric calculated on the user-document graph
        inspired by paper: https://onlinelibrary.wiley.com/doi/10.1002/asi.20591
        """
        if not self.nx_graph.has_node(node1) or not self.nx_graph.has_node(node2):
            return 0
        res = 0
        neighbors_node1 = self.neighbors(node1)
        for node in neighbors_node1:
            intersection = nx.common_neighbors(self.nx_graph, node, node2)
            res += len(intersection) * self.get_weight(node1, node) / self.degree(node)
        neighbors_node2 = self.neighbors(node2)
        for node in neighbors_node2:
            intersection = nx.common_neighbors(self.nx_graph, node, node1)
            res += len(intersection) * self.get_weight(node2, node) / self.degree(node)
        return res

    def adamic_adar_metric(self, node1, targets):
        """Wrapper for the Adamic-Adar metric computation meant to be run on an array of target nodes."""
        return np.array([self.adamic_adar_single(node1, node2) for node2 in targets])

    def adamic_adar_single(self, node1, node2):
        """
        Adamic-Adar similarity metrics defined in https://www.sciencedirect.com/science/article/pii/S0378873303000091?via%3Dihub.
        It has been adapted to the bipartite aspect of the graph in the same way as common neighbors.
        """
        if not self.nx_graph.has_node(node1) or not self.nx_graph.has_node(node2):
            return 0
        res = 0
        neighbors_node1 = self.neighbors(node1)
        for node in neighbors_node1:
            tmp = 0
            for z in nx.common_neighbors(self.nx_graph, node, node2):
                tmp += 1 / math.log(self.degree(z) + EPSILON)
            res += tmp * self.get_weight(node1, node) / self.degree(node)
        neighbors_node2 = self.neighbors(node2)
        for node in neighbors_node2:
            tmp = 0
            for z in nx.common_neighbors(self.nx_graph, node, node1):
                tmp += 1 / math.log(self.degree(z) + EPSILON)
            res += tmp * self.get_weight(node2, node) / self.degree(node)
        return res

    def rooted_page_rank(self, root_node, target_nodes):
        """Compute a page rank metric where all the jumps are made back to the root node."""
        if not self.nx_graph.has_node(root_node):
            return np.zeros(shape=(len(target_nodes)))
        pagerank_scores = nx.pagerank(
            self.nx_graph, weight="weight", personalization={root_node: 1}
        )
        return np.array(
            [pagerank_scores.get(node, 0) + EPSILON for node in target_nodes]
        )

    def prop_flow(self, source_node, targets, max_length: int = 5):
        """
        Prop flow similarity metric proposed by Lichtenwalter et al. in https://doi.org/10.1145/1835804.1835837.
        """
        if not self.nx_graph.has_node(source_node):
            return np.zeros(shape=(len(targets)))
        visit_probability = {source_node: 1.0}
        for step in range(max_length):
            next_visit_probability = {}
            for current_node, prob in visit_probability.items():
                neighbors = self.neighbors(current_node)
                if not neighbors:
                    continue
                transfer_prob = prob / len(neighbors)
                for neighbor in neighbors:
                    if neighbor not in next_visit_probability:
                        next_visit_probability[neighbor] = 0
                    next_visit_probability[neighbor] += transfer_prob

            visit_probability = next_visit_probability

        return np.array([visit_probability.get(node, 0) for node in targets])

    def display_stats(self):
        """
        Display information about the graph.

        The code for the plots is adapted from networkx documentation: https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html#sphx-glr-auto-examples-drawing-plot-degree-py
        """
        G = self.nx_graph
        print(f"Graph {self.name}:")
        print(f"|N| = {len(G.nodes)}\t |E| = {len(G.edges)}")
        print(f"connected components: {nx.number_connected_components(G)}")
        giant_component = sorted(list(nx.connected_components(G)))[0]
        print(f"giant component: {len(giant_component)}")
        giant_component_graph = G.subgraph(giant_component)
        print(
            f"diameter(lower bound): {nx.algorithms.approximation.diameter(giant_component_graph)}"
        )
        giant_component_subset = random.choices(list(giant_component), k=30)
        avg = 0
        for node in giant_component_subset:
            avg += np.average(list(nx.shortest_path_length(G, node).values())) / 30
        print(f"average distance: {avg}")

        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        fig = plt.figure("Degree of a random graph", figsize=(8, 4))
        axgrid = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(axgrid[:, :1])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[:, 1:])
        ax2.hist(degree_sequence)
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    user_pages = Graph(0, 5, False, name="user-pages")
    user_pages.display_stats()
