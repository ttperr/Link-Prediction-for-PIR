import csv
import numpy as np
from tqdm import tqdm
from graph import Graph
from Reranker import Reranker


class mlRanking:

    def __init__(self, dataset):
        self.dataset = dataset
        self.reranker = Reranker(dataset)

    def edge_features(self, query, user, documents):
        return self.reranker.graph_features(query, user, documents)
